from fastapi import APIRouter
from app.services.data_loader import load_data

router = APIRouter(prefix="/colours", tags=["colours"])


@router.get("/lagos-fw")
def overall_colours():
    data = load_data("runway_colours.json")
    return data

@router.get("/brand/{brand_name}")
def get_brand_colours(brand_name: str):
    filename = f"runway_colours.json"
    data = load_data(filename)
    new_data = [item for item in data if item.get('brand').lower() == brand_name.lower()]
    return {"brand_colours": new_data}

@router.get("/brand/{brand_name}/top5")
def get_brand_top5_colours(brand_name: str):
    filename = f"{brand_name}_colours.json"
    data = load_data(filename)
    sorted_colours = sorted(
        data,
        key=lambda x: x["count"],
        reverse=True)
    
    top5 = sorted_colours[:5]
    names = [item["colour"] for item in top5]
    percentages_sentence = [
        f"{item['colour']} is seen in {item['percentage']:.2f}% of the collection." for item in top5]
    
    return {"brand_top5_colours": names,
            "percentages": percentages_sentence}

@router.get("/brand/{brand_name}/dominant")
def dominant_colour(brand_name: str):
    filename = f"{brand_name}_colours.json"
    data = load_data(filename)
    dominant = max(data, key=lambda x: x["count"])

    return {
        "dominant_colour": dominant["colour"],
        "percentage": dominant["percentage"],
        "insight": f"{dominant['colour']} is the dominant colour appearing in {dominant['percentage']:.2f}% of the collection."}