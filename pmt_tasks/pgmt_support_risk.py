"""T1 v3: contact-plane / anchored-terrain geometry, with a fixed 25-point foot.

Pure tensor geometry. Positive rates are costs; RewardManager supplies the
negative weight and dt. Neither reference labels nor touchdown counts enter T1.
"""
import math
import torch
from pmt_tasks.pgmt_math import rotate


SUPPORT_RISK_CONFIG = dict(
    version='support_risk_v3', footprint_version='g1_imported_capsules_v1',
    support_tolerance_m=.015, edge_tolerance_m=.005,
    maximum_slope_deg=40., fit_tolerance_m=.008, anchor_gap_m=.03,
    deduplicate_m=.0001, minimum_spread_m=.001, maximum_condition=100.,
    minimum_coverage=.7, minimum_margin_m=.02, rms_allowance_m=.005,
    rms_scale_m=.03, coverage_weight=.5, edge_weight=.3, rms_weight=.2,
    edge_grid_spacing_m=.012,
)

# Isaac's URDF importer replaces these cylinders with CAPSULES. End caps
# extend the longitudinal footprint by 10 mm at heel and toe. Parameters are
# audited against the actual native_robot/main.usd collision prims, not visuals.
FOOT_CAPSULES = ((.05, .10, -.026, .010), (-.044, .123, -.018, .008),
                 (-.052, .130, -.010, .010), (-.054, .132, 0., .010),
                 (-.052, .130, .010, .010), (-.044, .123, .018, .008),
                 (.05, .10, .026, .010))
FOOT_OUTLINE_SEGMENTS = 16
FOOT_OUTLINE_ERROR_M = .01*(1-math.cos(math.pi/FOOT_OUTLINE_SEGMENTS))


def _capsule_hull():
    points = sorted(set((x+r*math.cos(2*math.pi*k/FOOT_OUTLINE_SEGMENTS),
                         y+r*math.sin(2*math.pi*k/FOOT_OUTLINE_SEGMENTS))
                        for a, b, y, r in FOOT_CAPSULES for x in (a, b)
                        for k in range(FOOT_OUTLINE_SEGMENTS)))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def half(points):
        hull = []
        for point in points:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 1e-15:
                hull.pop()
            hull.append(point)
        return hull
    return tuple(half(points)[:-1]+half(points[::-1])[:-1])


FOOT_HULL = _capsule_hull()


def validate_support_risk_config(c):
    if (set(c) != set(SUPPORT_RISK_CONFIG) or c['version'] != 'support_risk_v3'
            or c['footprint_version'] != 'g1_imported_capsules_v1'):
        raise ValueError('T1 geometry configuration/version does not match')
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
               and (v >= 0 if k.endswith('_weight') else v > 0)
               for k, v in c.items() if k not in ('version', 'footprint_version')):
        raise ValueError('T1 scales must be positive and weights nonnegative, all finite')
    if (c['maximum_slope_deg'] >= 80 or c['minimum_coverage'] > 1
            or abs(sum(c[k] for k in ('coverage_weight', 'edge_weight', 'rms_weight')) - 1) > 1e-6):
        raise ValueError('Invalid T1 slope, coverage, or cost weights')


def foot_sampling(*, device, dtype):
    """Five transverse rows, five longitudinal samples, fixed area quadrature.

    Row endpoints lie on the convex collision envelope. Trapezoidal area weights
    retain full-foot area; removing fit outliers never renormalizes these weights.
    """
    y = torch.tensor([-.036, -.018, 0., .018, .036], device=device, dtype=dtype)
    bounds = []
    for row_y in (-.036, -.018, 0., .018, .036):
        intersections = []
        for a, b in zip(FOOT_HULL, FOOT_HULL[1:]+FOOT_HULL[:1]):
            if min(a[1], b[1])-1e-12 <= row_y <= max(a[1], b[1])+1e-12:
                if abs(b[1]-a[1]) < 1e-12:
                    intersections.extend((a[0], b[0]))
                else:
                    t = min(1., max(0., (row_y-a[1])/(b[1]-a[1])))
                    intersections.append(a[0]+t*(b[0]-a[0]))
        bounds.append((min(intersections), max(intersections)))
    xmin, xmax = y.new_tensor(bounds).unbind(-1)
    u = torch.linspace(0, 1, 5, device=device, dtype=dtype)
    x = xmin[:, None] + (xmax-xmin)[:, None]*u
    local = torch.stack((x, y[:, None].expand_as(x), torch.full_like(x, -.035)), -1).reshape(25, 3)
    trapezoid = y.new_tensor([.5, 1, 1, 1, .5])
    weights = ((xmax-xmin)[:, None] * trapezoid[:, None] * trapezoid[None]).reshape(25)
    return local, weights / weights.sum()


def _query(query, xy):
    result = query(xy.reshape(-1, 2))
    h = result['height'].reshape(xy.shape[:-1])
    valid = result['valid'].reshape_as(h) & torch.isfinite(h)
    return torch.where(valid, h, 0.), valid


def _fit(points, mask, config, anchor=None):
    """Centered SVD avoids world-origin cancellation and singular normal solves."""
    safe = torch.where(mask[..., None], points, 0.)
    count = mask.sum(-1)
    center = safe.sum(-2) / count.clamp_min(1)[..., None] if anchor is None else anchor
    x = torch.where(mask[..., None], safe[..., :2] - center[:, None, :2], 0.)
    z = torch.where(mask, safe[..., 2] - center[:, None, 2], 0.)
    # A one-sided Jacobi SVD for exactly TWO centered columns. The right
    # rotation orthogonalizes X, so U*S=X*V and the two singular values are
    # simply column norms. Avoid thousands of separate cuSOLVER calls for tall
    # 64x2 matrices, and never invert the ill-conditioned normal equations.
    xx, xy, yy = x[..., 0].square().sum(-1), (x[..., 0]*x[..., 1]).sum(-1), x[..., 1].square().sum(-1)
    theta = .5*torch.atan2(2*xy, xx-yy)
    cosine, sine = theta.cos(), theta.sin()
    first = cosine[:, None]*x[..., 0] + sine[:, None]*x[..., 1]
    second = -sine[:, None]*x[..., 0] + cosine[:, None]*x[..., 1]
    singular_sq = torch.stack((first.square().sum(-1), second.square().sum(-1)), -1)
    singular = singular_sq.sqrt()
    coefficients = torch.stack(((first*z).sum(-1), (second*z).sum(-1)), -1) / singular_sq.clamp_min(1e-16)
    slope = torch.stack((cosine*coefficients[:, 0]-sine*coefficients[:, 1],
                         sine*coefficients[:, 0]+cosine*coefficients[:, 1]), -1)
    residual = torch.where(mask, z - (x * slope[:, None]).sum(-1), 0.)
    rms = (residual.square().sum(-1)/count.clamp_min(1)).sqrt()
    spread = singular[..., 1] / count.clamp_min(1).sqrt()
    valid = ((count >= 3) & (spread >= config['minimum_spread_m'])
             & (singular[..., 0] <= config['maximum_condition']*singular[..., 1])
             & (rms <= config['fit_tolerance_m'])
             & (slope.norm(dim=-1) <= math.tan(math.radians(config['maximum_slope_deg']))))
    return center, slope, valid


def _connected(height, xy, valid, seed, rows, cols, config, extra=None):
    """Four-neighbour continuity; graph distances need at most rows*cols hops."""
    h = height.reshape(-1, rows, cols)
    p = xy.reshape(-1, rows, cols, 2)
    ok = valid.reshape(-1, rows, cols)
    if extra is not None:
        ok = ok & extra.reshape_as(ok)
    slope = math.tan(math.radians(config['maximum_slope_deg']))
    horizontal = (ok[:, :, 1:] & ok[:, :, :-1]
                  & ((h[:, :, 1:]-h[:, :, :-1]).abs()
                     <= config['edge_tolerance_m'] + slope*(p[:, :, 1:]-p[:, :, :-1]).norm(dim=-1)))
    vertical = (ok[:, 1:] & ok[:, :-1]
                & ((h[:, 1:]-h[:, :-1]).abs()
                   <= config['edge_tolerance_m'] + slope*(p[:, 1:]-p[:, :-1]).norm(dim=-1)))
    reached = torch.zeros_like(valid).scatter(-1, seed[:, None], True).reshape_as(ok) & ok
    # Full connectivity, including detours around holes. No CPU convergence sync.
    for _ in range(rows*cols-1):
        spread = reached.clone()
        spread[:, :, 1:] |= reached[:, :, :-1] & horizontal
        spread[:, :, :-1] |= reached[:, :, 1:] & horizontal
        spread[:, 1:] |= reached[:, :-1] & vertical
        spread[:, :-1] |= reached[:, 1:] & vertical
        reached = spread
    return reached.reshape_as(valid)


def _signed_polygon_distance(points, hull):
    """Positive outside a convex CCW footprint, negative inside (world XY)."""
    start, end = hull, hull.roll(-1, -2)
    edge = end-start
    relative = points[:, :, None]-start[:, None]
    t = (relative*edge[:, None]).sum(-1)/edge.square().sum(-1)[:, None].clamp_min(1e-12)
    distance = (relative-t.clamp(0, 1)[..., None]*edge[:, None]).norm(dim=-1).amin(-1)
    cross = edge[:, None, :, 0]*relative[..., 1] - edge[:, None, :, 1]*relative[..., 0]
    inside = (cross >= -1e-7).all(-1) | (cross <= 1e-7).all(-1)
    return torch.where(inside, -distance, distance)


class SupportRiskGeometry:
    def __init__(self, *, device, dtype=torch.float32, config=None):
        self.config = dict(SUPPORT_RISK_CONFIG if config is None else config)
        validate_support_risk_config(self.config)
        self.local, self.weights = foot_sampling(device=device, dtype=dtype)
        hull = torch.tensor(FOOT_HULL, device=device, dtype=dtype)
        self.hull = torch.cat((hull, torch.full_like(hull[:, :1], -.035)), -1)
        # Outside the whole envelope by margin + a full grid cell. Window edges
        # are not hazards: only measured unsupported/unknown terrain is unsafe.
        spacing = self.config['edge_grid_spacing_m']
        pad = self.config['minimum_margin_m'] + 2*spacing
        diameter = max(math.dist(a, b) for a in FOOT_HULL for b in FOOT_HULL) + 2*FOOT_OUTLINE_ERROR_M
        self.cols = math.ceil((diameter + 2*pad)/spacing)+1
        self.rows = math.ceil((.072 + 2*pad)/spacing)+1
        self.outer_pad = pad
        x = torch.linspace(0, 1, self.cols, device=device, dtype=dtype)
        y = torch.linspace(0, 1, self.rows, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.outer_uv = torch.stack((xx, yy), -1).reshape(-1, 2)

    @torch.no_grad()
    def evaluate(self, support, support_valid, foot_pos, foot_quat, points, point_valid, overflow, query):
        c, n = self.config, len(foot_pos)
        pos, quat = foot_pos.reshape(-1, 3), foot_quat.reshape(-1, 4)
        pose_ok = torch.isfinite(pos).all(-1) & torch.isfinite(quat).all(-1) & (quat.norm(dim=-1) > 1e-6)
        pos = torch.nan_to_num(pos)
        quat = torch.nan_to_num(quat) / torch.nan_to_num(quat.norm(dim=-1, keepdim=True)).clamp_min(1e-6)
        def world(local):
            return pos[:, None] + rotate(quat[:, None].expand(-1, len(local), -1), local[None].expand(len(pos), -1, -1))
        foot, hull = world(self.local), world(self.hull)
        # Surround the ACTUAL projected footprint in a horizontal basis. Rotating
        # a padded sole grid in 3D would shrink its XY margin during heel/toe roll
        # and incorrectly charge risk on flat terrain.
        forward = rotate(quat, pos.new_tensor([1., 0., 0.]).expand_as(pos))[..., :2]
        norm = forward.norm(dim=-1, keepdim=True)
        axis_x = torch.where(norm > 1e-6, forward/norm.clamp_min(1e-6), forward.new_tensor([1., 0.]))
        axis_y = torch.stack((-axis_x[:, 1], axis_x[:, 0]), -1)
        relative = hull[..., :2]-pos[:, None, :2]
        local_hull = torch.stack(((relative*axis_x[:, None]).sum(-1), (relative*axis_y[:, None]).sum(-1)), -1)
        lo, hi = local_hull.amin(-2)-self.outer_pad, local_hull.amax(-2)+self.outer_pad
        local_outer = lo[:, None]+self.outer_uv[None]*(hi-lo)[:, None]
        outer_xy = pos[:, None, :2]+local_outer[..., :1]*axis_x[:, None]+local_outer[..., 1:]*axis_y[:, None]
        cell_radius = .5*((hi-lo)/pos.new_tensor([self.cols-1, self.rows-1])).norm(dim=-1)
        height, valid = _query(query, foot[..., :2])
        terrain = torch.cat((foot[..., :2], height[..., None]), -1)
        cloud = points.reshape(len(pos), -1, 3)
        mask = point_valid.reshape(cloud.shape[:-1]) & torch.isfinite(cloud).all(-1)
        cloud = torch.nan_to_num(cloud)
        distance = (cloud[:, :, None]-cloud[:, None]).square().sum(-1)
        earlier = torch.ones(cloud.shape[1], cloud.shape[1], device=pos.device, dtype=torch.bool).tril(-1)
        duplicate = ((distance <= c['deduplicate_m']**2) & earlier & mask[:, None]).any(-1)
        mask &= ~duplicate
        count = mask.sum(-1)
        center, slope, direct = _fit(cloud, mask, c)
        direct &= ~overflow.reshape(-1)
        # Verify the bridge between actual contacts: a plane through points on
        # different stair treads must not be mistaken for a continuous ramp.
        anchor_index = mask.to(torch.int32).argmax(-1)
        anchor = cloud.gather(1, anchor_index[:, None, None].expand(-1, 1, 3)).squeeze(1)
        fractions = pos.new_tensor([.25, .5, .75])
        bridge = anchor[:, None, None, :2] + fractions[None, None, :, None]*(cloud[:, :, None, :2]-anchor[:, None, None, :2])
        bridge_h, bridge_ok = _query(query, bridge)
        bridge_plane = center[:, None, None, 2] + ((bridge-center[:, None, None, :2])*slope[:, None, None]).sum(-1)
        direct &= ((bridge_ok & ((bridge_h-bridge_plane).abs() <= c['fit_tolerance_m'])) | ~mask[..., None]).all(dim=(-1, -2))
        # A real contact anchors fallback at terrain height at that contact XY.
        anchor_h, anchor_ok = _query(query, anchor[:, None, :2])
        has_contact = (count > 0) & anchor_ok[:, 0] & ((anchor_h[:, 0]-anchor[:, 2]).abs() <= c['fit_tolerance_m'])
        gap = (height-foot[..., 2]).abs()
        proxy = torch.where(valid, gap, float('inf')).argmin(-1)
        proxy_point = terrain.gather(1, proxy[:, None, None].expand(-1, 1, 3)).squeeze(1)
        anchor = torch.where((count > 0)[:, None], anchor, proxy_point)
        anchor_valid = has_contact | ((count == 0) & valid.gather(1, proxy[:, None]).squeeze(-1)
                                      & (gap.gather(1, proxy[:, None]).squeeze(-1) <= c['anchor_gap_m']))
        nearest = (foot[..., :2]-anchor[:, None, :2]).norm(dim=-1)
        seed_ok = valid & ((height-anchor[:, None, 2]).abs() <= c['edge_tolerance_m']
                          + math.tan(math.radians(c['maximum_slope_deg']))*nearest)
        seed = torch.where(seed_ok, nearest, float('inf')).argmin(-1)
        component = _connected(height, foot[..., :2], valid, seed, 5, 5, c)
        component &= seed_ok.gather(1, seed[:, None]) & anchor_valid[:, None]
        fallback_center, fallback_slope, fallback = _fit(terrain, component, c, anchor=anchor)
        fallback &= anchor_valid
        center = torch.where(direct[:, None], center, fallback_center)
        slope = torch.where(direct[:, None], slope, fallback_slope)
        plane_valid = (direct | fallback) & pose_ok
        delta = height - (center[:, None, 2]+((foot[..., :2]-center[:, None, :2])*slope[:, None]).sum(-1))
        near_plane = delta.abs() <= c['support_tolerance_m']
        connected = _connected(height, foot[..., :2], valid, seed, 5, 5, c, extra=near_plane)
        rms = (self.weights*delta.square()).sum(-1).sqrt()
        coverage = (self.weights*(connected & near_plane)).sum(-1)
        # Peripheral queries do not enter the 25-point RMS / coverage.
        oh, ov = _query(query, outer_xy)
        od = oh-(center[:, None, 2]+((outer_xy-center[:, None, :2])*slope[:, None]).sum(-1))
        # Plane compatibility plus continuity of the anchored footprint. An
        # unknown peripheral sample is conservatively an unconfirmed margin.
        safe = ov & (od.abs() <= c['support_tolerance_m'])
        distance_to_foot = _signed_polygon_distance(outer_xy, hull[..., :2])
        margin = torch.where(~safe, distance_to_foot-cell_radius[:, None]-FOOT_OUTLINE_ERROR_M, float('inf')).amin(-1)
        margin = margin.clamp_max(c['minimum_margin_m'])
        # Confirm sufficient world-XY coverage; never count query-window edges
        # as actual terrain edges.
        boundary = torch.zeros(self.rows, self.cols, dtype=torch.bool, device=pos.device)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        confirmed = (distance_to_foot[:, boundary.reshape(-1)].amin(-1)-cell_radius-FOOT_OUTLINE_ERROR_M >= c['minimum_margin_m'])
        ca = ((c['minimum_coverage']-coverage)/c['minimum_coverage']).clamp(0, 1)
        ce = ((c['minimum_margin_m']-margin)/c['minimum_margin_m']).clamp(0, 1)
        ce = torch.where(confirmed, ce, 1.)
        cr = ((rms-c['rms_allowance_m'])/c['rms_scale_m']).clamp(0, 1)
        good = plane_valid & valid.all(-1)
        cost = c['coverage_weight']*ca + c['edge_weight']*ce + c['rms_weight']*cr
        cost = torch.where(good, cost, 1.).reshape(n, 2)
        # Unknown support sensor data is not permission to report zero risk.
        active = support | ~support_valid
        result = dict(rate=(cost*active).sum(-1)/2, per_foot_cost=cost,
                      support=support.clone(), support_valid=support_valid.clone())
        for key, value in dict(valid=good, plane_valid=plane_valid,
                plane_source=torch.where(direct, 1, torch.where(fallback & pose_ok, 2, 0)),
                contact_count=count, contact_overflow=overflow.reshape(-1),
                rms=rms, coverage=coverage, margin=margin, margin_confirmed=confirmed,
                coverage_cost=ca, edge_cost=ce, rms_cost=cr,
                plane_origin=center, plane_slope=slope, query_xy_w=foot[..., :2],
                height_w=height, point_valid=valid).items():
            result[key] = value.reshape(n, 2, *value.shape[1:])
        return result
