from fastapi import APIRouter
from app.services.data_loader import load_data

router = APIRouter(prefix="/insights", tags=["insights"])

@router.get("/collection-summary")
def get_collection_summary():
    colour_data = load_data("runway_colours.json")
    # fabric_data = load_data("runway_fabrics.json")

    summary = {
        "top_colours": colour_data.get("top5_colours", []),
        "colour_frequency": colour_data.get("colour_frequency", {})
        #"top_fabrics": fabric_data.get("top_fabrics", []),
        #"fabric_frequency": fabric_data.get("fabric_frequency", {})
    }
    
    return {"collection_summary": summary}