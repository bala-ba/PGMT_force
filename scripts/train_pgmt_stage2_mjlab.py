"""PGMT Stage2 entry point for MuJoCo Warp; Isaac's train_pgmt_stage2.py is unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_pgmt_mjlab import main

if __name__ == "__main__":
    main(stage=2)
