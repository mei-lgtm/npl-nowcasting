"""Static demo UI served by FastAPI when Next.js is unavailable."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

static_dir = Path(__file__).resolve().parents[1] / "static"
router = APIRouter()


def mount_static(app):
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def spa_index():
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse("<h1>NPL Nowcasting API</h1><p>Open <a href='/docs'>/docs</a></p>")
