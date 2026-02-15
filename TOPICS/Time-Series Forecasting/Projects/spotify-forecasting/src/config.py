# =======================
# This is a large dataset, so I need to create a shortcut to my local downloads folder
# This avoids typing the full path everytime

import os
from pathlib import Path

# 1. Project Root
PROJ_ROOT = Path(__file__).resolve().parents[1]

# 2. Data Root (This is where the Spotify CSV is located - local as its large)
DATA_RAW = Path("/Users/joelangstaff/Downloads")

# 3. Validation
if not DATA_RAW.exists():
    print(f"WARNING: Data path not found at {DATA_RAW}")
else:
    print(f"Config loaded. Pointing to raw data at: {DATA_RAW}")





