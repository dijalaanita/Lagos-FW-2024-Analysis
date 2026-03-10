from app.services.data_loader import load_data

data = load_data("babayo_colours.json")

for item in data:
    print(f"{item['colour']} → {item['percentage']}%")