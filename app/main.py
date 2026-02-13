from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Automated Background Replacement API",
    description="Deep Learning-powered background removal and replacement",
    version="1.0.0",
)

app.include_router(router)
