from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI


def build_artifact_cleanup_lifespan(
    *,
    cleanup_func: Callable[..., dict[str, int]],
    logger: logging.Logger,
    temp_dir_getter: Callable[[], Path],
    images_dir_getter: Callable[[], Path],
    interval_seconds_getter: Callable[[], int],
):
    async def _artifact_cleanup_loop() -> None:
        while True:
            await asyncio.sleep(interval_seconds_getter())
            try:
                summary = cleanup_func(temp_dir=temp_dir_getter(), images_dir=images_dir_getter())
                logger.info("Expired artifacts cleaned", extra=summary)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Artifact cleanup loop failed")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        summary = cleanup_func(temp_dir=temp_dir_getter(), images_dir=images_dir_getter())
        logger.info("Startup artifact cleanup completed", extra=summary)
        task = asyncio.create_task(_artifact_cleanup_loop())
        app.state.artifact_cleanup_task = task
        try:
            yield
        finally:
            task = getattr(app.state, "artifact_cleanup_task", None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return lifespan
