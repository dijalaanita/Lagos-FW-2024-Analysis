from fastapi import FastAPI
from app.routers import colours, insights, fabrics

app = FastAPI(
    title="Lagos Fashion Week 2025 Analysis API",
    description="API for accessing insights and data from Lagos Fashion Week 2025 analysis.",
    version="1.0.0"
                )

app.include_router(colours.router)
app.include_router(fabrics.router)
app.include_router(insights.router)