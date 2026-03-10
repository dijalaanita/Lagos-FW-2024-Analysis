from fastapi import APIRouter
from app.services.data_loader import load_data

router = APIRouter(prefix="/fabrics", tags=["Fabrics"])

@router.get("/top5")
def get_top_fabrics():
    data = load_data()

    sorted_fabrics = sorted(
        data.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {"top_fabrics": sorted_fabrics[:5]}


@router.get("/frequency")
def get_fabric_frequency():
    return load_data()