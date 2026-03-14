import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "outputs" / "stats" / "JSON"

def load_data(file):
    file_path = DATA_DIR / file
    print(f"DEBUG: Looking for file at: {file_path.absolute()}") # <--- ADD THIS
    # 1. Check if file exists before opening
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return [] # Return empty string so 'for item in data' doesn't crash

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Corrupt JSON file: {file_path}")
        return []
    except Exception as e:
        print(f"🔥 Unexpected error: {e}")
        return []