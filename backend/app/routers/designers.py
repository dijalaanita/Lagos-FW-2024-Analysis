from fastapi import APIRouter

router = APIRouter(prefix="/designers", tags=["designers"])

@router.get("/{designer_name}")
def get_designer_info(designer_name: str):
    # database query or data retrieval logic to get designer info
    return {"designer_name": designer_name,
             "dominant_colour": ["Red", "Black"],
             "dominant_fabric": ["Silk", "Cotton"],
             }