"""NPL Nowcasting Intelligence System — FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.core.config import settings
from app.services.bootstrap import bootstrap_demo_if_needed
from app.static_mount import mount_static


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_demo_if_needed()
    yield


app = FastAPI(
    title="NPL Nowcasting Intelligence System",
    description=(
        "AI-powered Non-Performing Loan nowcasting with news sentiment analytics. "
        "Distinguishes Actual NPL, Nowcast NPL, and Forecast NPL. "
        "Demo mode uses clearly labeled synthetic data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
mount_static(app)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": settings.app_mode,
        "demo": settings.app_mode == "demo",
    }
