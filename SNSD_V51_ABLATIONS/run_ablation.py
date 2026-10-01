"""Run one learned or Oracle ablation cell against the frozen final core."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
FINAL_SRC = ROOT.parent / "SNSD_V51_FINAL" / "src"
ABLATION_SRC = ROOT / "src"
for path in (FINAL_SRC, ABLATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from snsd_v51_ablations.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
