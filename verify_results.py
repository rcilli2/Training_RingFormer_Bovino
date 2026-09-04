from pathlib import Path

import matplotlib
import numpy
import pandas
import sklearn


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / "data" / "lobo_predictions.csv",
    ROOT / "data" / "loco_mc_dropout_predictions.csv",
    ROOT / "data" / "buffer_uq_method_metrics.csv",
    ROOT / "data" / "feature_perturbation_all_methods.csv",
    ROOT / "data" / "representation_lobo_metrics.csv",
    ROOT / "data" / "representation_loco_metrics.csv",
]

missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
if missing:
    raise SystemExit(f"Missing packaged data: {missing}")

print("Saved-results verification passed")
print(
    f"numpy={numpy.__version__}, pandas={pandas.__version__}, "
    f"matplotlib={matplotlib.__version__}, sklearn={sklearn.__version__}"
)
