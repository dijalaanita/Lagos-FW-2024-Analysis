import json
from pathlib import Path

DATA_DIR = Path(r"C:\Users\User\OneDrive\Desktop\Lagos-FW-2024-Analysis\Lagos-FW-2024-Analysis-1\outputs\stats\JSON")

def load_data(file):
    with open(DATA_DIR / file, 'r') as f:
        return json.load(f)