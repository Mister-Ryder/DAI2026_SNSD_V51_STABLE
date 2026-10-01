"""Train the message-policy warm-start used by the selected strict model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from snsd_v51.message_warmstart_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
