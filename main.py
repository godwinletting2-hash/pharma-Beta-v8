"""
main.py
-------
PharmaScanKE — Application entry point for Railway / Replit deployment.

Run locally:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
Railway:      Procfile → uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routes import router as api_router
from core.config import settings
from core.database import init_db


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database schema on startup."""
    await init_db()
    yield


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])

# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_credentials=True is incompatible with allow_origins=["*"] per the Fetch
# spec — browsers will block preflight.  We allow all origins without credentials
# (safe for a public study-tool with no cookies / auth headers).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)

# ── Static mounts ─────────────────────────────────────────────────────────────
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/uploaded_notes",
    StaticFiles(directory=str(settings.UPLOAD_DIR)),
    name="uploaded_notes",
)
app.mount(
    "/static",
    StaticFiles(directory=str(settings.STATIC_DIR)),
    name="static",
)

# ── Template engine ───────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# ── API router ────────────────────────────────────────────────────────────────
app.include_router(api_router)


# ── Health check (used by Railway) ────────────────────────────────────────────
@app.get("/healthz", include_in_schema=False)
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": settings.APP_TITLE, "version": settings.APP_VERSION})


# ── Root route ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {
        "request": request,
        "whatsapp_url": settings.WHATSAPP_CHANNEL_URL.strip(),
        "telegram_url": settings.TELEGRAM_CHANNEL_URL.strip(),
    })


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
