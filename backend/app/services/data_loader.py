import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "outputs" / "stats" / "JSON"

def load_data(file):
    file_path = DATA_DIR / file
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"🔥 Unexpected error loading JSON: {e}")
        return None