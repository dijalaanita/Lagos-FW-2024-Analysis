from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import colours, insights, fabrics

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

app.include_router(colours.router)
app.include_router(fabrics.router)
app.include_router(insights.router)