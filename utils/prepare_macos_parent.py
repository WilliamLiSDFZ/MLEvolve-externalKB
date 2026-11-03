"""CLI for Mac parent preparation; only --execute starts calibration training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.replay_macos.preparation import main


if __name__ == "__main__":
    main()
