from fastapi import APIRouter
from app.services.data_loader import load_data

router = APIRouter(prefix="/colours", tags=["colours"])

@router.get("/lagosfw25")
def get_lagosfw25_colours():
    data = load_data("lagosfw25_colours.json")
    return data

