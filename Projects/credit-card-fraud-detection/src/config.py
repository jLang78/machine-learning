from pathlib import Path

# This gets the path of the file you are currently in (config.py)
# .parents[1] goes up two levels: src -> credit-card-fraud-detection
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJ_ROOT / "data" / "raw"
DATA_PROCESSED = PROJ_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2

