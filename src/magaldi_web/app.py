"""FastAPI application for Magaldi Web UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from magaldi_web.routes import admin, dashboard, elements, repos, search, vectormap
from shared.config import MagaldiConfig, get_config, load_config

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("Starting Magaldi Web UI...")
    try:
        config = get_config()
    except RuntimeError:
        config = load_config()
    logger.info(f"Web server configured on {config.web.host}:{config.web.port}")
    yield
    logger.info("Shutting down Magaldi Web UI...")


def create_app(config: MagaldiConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional configuration. If not provided, loads from default.

    Returns:
        Configured FastAPI application.
    """
    if config is None:
        try:
            config = get_config()
        except RuntimeError:
            config = load_config()

    app = FastAPI(
        title="Magaldi Web API",
        description="Code discovery engine for AI agents and developers",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])
    app.include_router(search.router, prefix="/api/v1", tags=["search"])
    app.include_router(repos.router, prefix="/api/v1", tags=["repositories"])
    app.include_router(elements.router, prefix="/api/v1", tags=["elements"])
    app.include_router(admin.router, prefix="/api/v1", tags=["admin"])
    app.include_router(vectormap.router, prefix="/api/v1", tags=["vectormap"])

    # Serve static files for frontend (if built)
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
        logger.info(f"Serving frontend from {frontend_dist}")

    return app


def run_server(
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    """Run the web server.

    Args:
        host: Host to bind to. Defaults to config value.
        port: Port to bind to. Defaults to config value.
        reload: Enable auto-reload for development.
    """
    import uvicorn

    try:
        config = get_config()
    except RuntimeError:
        config = load_config()

    host = host or config.web.host
    port = port or config.web.port

    logger.info(f"Starting Magaldi Web UI on http://{host}:{port}")
    logger.info(f"Auto-reload: {'enabled' if reload else 'disabled'}")

    uvicorn.run(
        "magaldi_web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        reload_dirs=["src"] if reload else None,
    )
