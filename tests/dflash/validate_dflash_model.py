"""Full-model baseline/Eager/Graph/prefix alignment, using local weights only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mtp"))
from validate_mtp_model import main

if __name__ == "__main__":
    sys.argv.extend(("--method", "dflash"))
    if "--report" not in sys.argv:
        sys.argv.extend(("--report", "artifacts/dflash/model_validation.json"))
    main()
