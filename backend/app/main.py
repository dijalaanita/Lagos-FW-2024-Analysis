from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import colours, insights, fabrics
import os
from pathlib import Path

app = FastAPI(
    title="Lagos Fashion Week 2025 Analysis API",
    description="API for accessing insights and data from Lagos Fashion Week 2025 analysis.",
    version="1.0.0"
                )

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    )
# read the brands from the runway directory
BASE_RUNWAY = Path(__file__).resolve().parent.parent.parent
RUNWAY = BASE_RUNWAY / "data" / "runway"

@app.get("/brands")
def get_brands():
    brands = []
    for file in os.listdir(RUNWAY):
        path = os.path.join(RUNWAY, file)

        if os.path.isdir(path):
            brands.append(file)
    return brands

app.include_router(colours.router)
app.include_router(fabrics.router)
app.include_router(insights.router)