"""Asynchronous file IO operations."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from loguru import logger

from core.io.serializers import BaseSerializer, SerializerFactory


class AsyncFileHandler:
    """Asynchronous file handler for non-blocking IO operations."""

    def __init__(self, max_workers: int = 4):
        """Initialize async file handler.

        Args:
            max_workers: Maximum number of worker threads
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_saves: Dict[str, asyncio.Task] = {}
        self._debounce_timers: Dict[str, asyncio.TimerHandle] = {}

    async def save(
        self,
        data: Dict[str, Any],
        filepath: Path,
        serializer: Optional[BaseSerializer] = None,
        indent: Optional[int] = 2,
    ) -> None:
        """Asynchronously save data to file.

        Args:
            data: Data to save
            filepath: Target file path
            serializer: Optional serializer to use
            indent: Indentation level for pretty printing
        """
        if serializer is None:
            serializer = SerializerFactory.get_serializer(filepath)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            partial(serializer.save, data, filepath, indent)
        )

    async def load(
        self,
        filepath: Path,
        serializer: Optional[BaseSerializer] = None,
    ) -> Dict[str, Any]:
        """Asynchronously load data from file.

        Args:
            filepath: Source file path
            serializer: Optional serializer to use

        Returns:
            Loaded data
        """
        if serializer is None:
            serializer = SerializerFactory.get_serializer(filepath)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(serializer.load, filepath)
        )

    def debounced_save(
        self,
        data: Dict[str, Any],
        filepath: Path,
        delay: float = 0.5,
        serializer: Optional[BaseSerializer] = None,
        indent: Optional[int] = 2,
    ) -> None:
        """Schedule a debounced save operation.

        Multiple calls within the delay period will only trigger one save.

        Args:
            data: Data to save
            filepath: Target file path
            delay: Debounce delay in seconds
            serializer: Optional serializer to use
            indent: Indentation level for pretty printing
        """
        key = str(filepath)

        # Cancel existing timer
        if key in self._debounce_timers:
            self._debounce_timers[key].cancel()

        # Create new timer
        loop = asyncio.get_event_loop()
        timer = loop.call_later(
            delay,
            lambda: asyncio.create_task(
                self._do_debounced_save(key, data, filepath, serializer, indent)
            )
        )
        self._debounce_timers[key] = timer

    async def _do_debounced_save(
        self,
        key: str,
        data: Dict[str, Any],
        filepath: Path,
        serializer: Optional[BaseSerializer],
        indent: Optional[int],
    ) -> None:
        """Execute debounced save."""
        try:
            await self.save(data, filepath, serializer, indent)
        finally:
            self._debounce_timers.pop(key, None)

    async def batch_save(
        self,
        items: list[tuple[Dict[str, Any], Path]],
        serializer: Optional[BaseSerializer] = None,
        indent: Optional[int] = 2,
    ) -> list[tuple[Path, Optional[Exception]]]:
        """Save multiple files concurrently.

        Args:
            items: List of (data, filepath) tuples
            serializer: Optional serializer to use
            indent: Indentation level for pretty printing

        Returns:
            List of (filepath, error) tuples
        """
        tasks = [
            self._save_with_error_handling(data, filepath, serializer, indent)
            for data, filepath in items
        ]
        return await asyncio.gather(*tasks)

    async def _save_with_error_handling(
        self,
        data: Dict[str, Any],
        filepath: Path,
        serializer: Optional[BaseSerializer],
        indent: Optional[int],
    ) -> tuple[Path, Optional[Exception]]:
        """Save with error handling."""
        try:
            await self.save(data, filepath, serializer, indent)
            return filepath, None
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            return filepath, e

    def shutdown(self) -> None:
        """Shutdown the executor and cancel pending operations."""
        # Cancel all debounce timers
        for timer in self._debounce_timers.values():
            timer.cancel()
        self._debounce_timers.clear()

        # Shutdown executor
        self._executor.shutdown(wait=True)
        logger.debug("AsyncFileHandler shutdown complete")
