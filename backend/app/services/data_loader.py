import json
from pathlib import Path

DATA_DIR = Path(r"C:\Users\User\OneDrive\Desktop\Lagos-FW-2024-Analysis\Lagos-FW-2024-Analysis-1\outputs\stats\JSON")

def load_data(file):
    with open(DATA_DIR / file, 'r') as f:
        return json.load(f)
    
def get_lagosfw25_colours():
    return load_data("runway_colours.json")

#def get_lagosfw25_fabrics():
    #return load_data("lagosfw25_fabrics.json")