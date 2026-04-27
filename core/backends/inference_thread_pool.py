"""Multi-threaded inference pool for video frame processing.

This module provides a thread pool for parallel frame interpolation,
with support for cancellation and graceful shutdown.
"""

import threading
import queue
import ctypes
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

import torch
import numpy as np
from loguru import logger


class TaskStatus(Enum):
    """Status of an inference task."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class InferenceTask:
    """A single inference task.
    
    Attributes:
        task_id: Unique task identifier
        frame0: First frame tensor [C, H, W]
        frame1: Second frame tensor [C, H, W]
        timestep: Interpolation timestep (0.0 to 1.0)
        callback: Optional callback for result
    """
    task_id: int
    frame0: torch.Tensor
    frame1: torch.Tensor
    timestep: float
    callback: Optional[Callable[[int, torch.Tensor], None]] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[torch.Tensor] = None
    error: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of an inference task.
    
    Attributes:
        task_id: Task identifier
        frame: Interpolated frame tensor [C, H, W]
        success: Whether inference succeeded
        error: Error message if failed
    """
    task_id: int
    frame: Optional[torch.Tensor]
    success: bool
    error: Optional[str] = None


class InferenceWorker(threading.Thread):
    """Worker thread for frame interpolation inference.
    
    Each worker runs in its own thread and processes tasks from the queue.
    """
    
    def __init__(
        self,
        worker_id: int,
        task_queue: queue.Queue,
        result_queue: queue.Queue,
        cancel_event: threading.Event,
        model_factory: Callable[[], Any],
        device: torch.device,
    ):
        """Initialize worker thread.
        
        Args:
            worker_id: Unique worker identifier
            task_queue: Queue for pending tasks
            result_queue: Queue for completed results
            cancel_event: Event to signal cancellation
            model_factory: Factory function to create model instance
            device: Device to run inference on
        """
        super().__init__(name=f"InferenceWorker-{worker_id}", daemon=True)
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.cancel_event = cancel_event
        self.model_factory = model_factory
        self.device = device
        self._model: Optional[Any] = None
        self._stop_event = threading.Event()
        
    def _init_model(self) -> bool:
        """Initialize the model for this worker.
        
        Returns:
            True if initialization succeeded
        """
        try:
            self._model = self.model_factory()
            if hasattr(self._model, 'to'):
                self._model.to(self.device)
            if hasattr(self._model, 'eval'):
                self._model.eval()
            logger.debug(f"Worker {self.worker_id}: Model initialized on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to initialize model: {e}")
            return False
    
    def run(self) -> None:
        """Main worker loop."""
        # Initialize model
        if not self._init_model():
            return
        
        logger.debug(f"Worker {self.worker_id}: Started")
        
        while not self._stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop_event
                task: Optional[InferenceTask] = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if task is None:  # Poison pill
                break
            
            # Check if cancelled before processing
            if self.cancel_event.is_set():
                task.status = TaskStatus.CANCELLED
                self.result_queue.put(InferenceResult(
                    task_id=task.task_id,
                    frame=None,
                    success=False,
                    error="Cancelled"
                ))
                self.task_queue.task_done()
                continue
            
            # Process task
            task.status = TaskStatus.RUNNING
            try:
                with torch.no_grad():
                    result = self._model.interpolate(
                        task.frame0,
                        task.frame1,
                        timestep=task.timestep,
                    )
                
                # Check if cancelled after inference
                if self.cancel_event.is_set():
                    task.status = TaskStatus.CANCELLED
                    self.result_queue.put(InferenceResult(
                        task_id=task.task_id,
                        frame=None,
                        success=False,
                        error="Cancelled"
                    ))
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result.frame
                    self.result_queue.put(InferenceResult(
                        task_id=task.task_id,
                        frame=result.frame,
                        success=True
                    ))
                    
                    # Call callback if provided
                    if task.callback:
                        try:
                            task.callback(task.task_id, result.frame)
                        except Exception as e:
                            logger.warning(f"Callback error: {e}")
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Inference error: {e}")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                self.result_queue.put(InferenceResult(
                    task_id=task.task_id,
                    frame=None,
                    success=False,
                    error=str(e)
                ))
            
            finally:
                self.task_queue.task_done()
        
        # Cleanup
        if self._model and hasattr(self._model, 'unload'):
            try:
                self._model.unload()
            except Exception:
                pass
        
        logger.debug(f"Worker {self.worker_id}: Stopped")
    
    def stop(self) -> None:
        """Signal the worker to stop gracefully."""
        self._stop_event.set()
    
    def force_terminate(self) -> None:
        """Force terminate the worker thread.
        
        WARNING: This is dangerous and should only be used as last resort.
        May leave resources in inconsistent state.
        """
        if self.is_alive():
            # Get thread ID
            thread_id = self.ident
            if thread_id:
                try:
                    # Use ctypes to raise exception in thread
                    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread_id),
                        ctypes.py_object(SystemExit)
                    )
                    if res > 1:
                        # If more than one thread affected, revert
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_long(thread_id), None
                        )
                        logger.warning(f"Worker {self.worker_id}: Failed to force terminate")
                    else:
                        logger.warning(f"Worker {self.worker_id}: Force terminated")
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error during force terminate: {e}")


class InferenceThreadPool:
    """Thread pool for parallel frame interpolation inference.
    
    Manages multiple worker threads that process interpolation tasks
    concurrently. Supports cancellation and graceful shutdown.
    
    Example:
        pool = InferenceThreadPool(
            num_workers=4,
            model_factory=lambda: create_model(),
            device=torch.device("cuda:0")
        )
        pool.start()
        
        # Submit tasks
        for i, (f0, f1) in enumerate(frame_pairs):
            pool.submit(InferenceTask(
                task_id=i,
                frame0=f0,
                frame1=f1,
                timestep=0.5
            ))
        
        # Get results
        results = pool.get_results(timeout=30.0)
        
        # Shutdown
        pool.shutdown()
    """
    
    def __init__(
        self,
        num_workers: int,
        model_factory: Callable[[], Any],
        device: torch.device,
        queue_size: int = 100,
    ):
        """Initialize the thread pool.
        
        Args:
            num_workers: Number of worker threads
            model_factory: Factory function to create model instances
            device: Device to run inference on
            queue_size: Maximum size of task queue
        """
        self.num_workers = num_workers
        self.model_factory = model_factory
        self.device = device
        self.queue_size = queue_size
        
        self.task_queue: queue.Queue[Optional[InferenceTask]] = queue.Queue(maxsize=queue_size)
        self.result_queue: queue.Queue[InferenceResult] = queue.Queue()
        self.cancel_event = threading.Event()
        
        self.workers: List[InferenceWorker] = []
        self._started = False
        self._shutdown = False
        
        self._task_counter = 0
        self._task_counter_lock = threading.Lock()
        
    def start(self) -> None:
        """Start all worker threads."""
        if self._started:
            return
        
        logger.info(f"Starting inference thread pool with {self.num_workers} workers")
        
        for i in range(self.num_workers):
            worker = InferenceWorker(
                worker_id=i,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                cancel_event=self.cancel_event,
                model_factory=self.model_factory,
                device=self.device,
            )
            worker.start()
            self.workers.append(worker)
        
        self._started = True
        self._shutdown = False
        logger.info("Inference thread pool started")
    
    def submit(self, task: InferenceTask) -> bool:
        """Submit a task to the pool.
        
        Args:
            task: Inference task to process
            
        Returns:
            True if task was submitted successfully
        """
        if self._shutdown or self.cancel_event.is_set():
            return False
        
        try:
            self.task_queue.put(task, timeout=1.0)
            return True
        except queue.Full:
            logger.warning("Task queue is full")
            return False
    
    def submit_batch(
        self,
        frames: List[torch.Tensor],
        multiplier: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[int]:
        """Submit a batch of interpolation tasks.
        
        Args:
            frames: List of frames [N, C, H, W]
            multiplier: Interpolation multiplier
            progress_callback: Optional progress callback (current, total)
            
        Returns:
            List of task IDs
        """
        task_ids = []
        n = len(frames)
        total_tasks = (n - 1) * (multiplier - 1)
        
        for i in range(n - 1):
            for j in range(1, multiplier):
                with self._task_counter_lock:
                    task_id = self._task_counter
                    self._task_counter += 1
                
                timestep = j / multiplier
                
                task = InferenceTask(
                    task_id=task_id,
                    frame0=frames[i],
                    frame1=frames[i + 1],
                    timestep=timestep,
                )
                
                if self.submit(task):
                    task_ids.append(task_id)
                
                if progress_callback:
                    progress_callback(len(task_ids), total_tasks)
        
        return task_ids
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[InferenceResult]:
        """Get a result from the result queue.
        
        Args:
            timeout: Timeout in seconds (None = block indefinitely)
            
        Returns:
            InferenceResult or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_results(
        self,
        expected_count: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[InferenceResult]:
        """Get all results from the result queue.
        
        Args:
            expected_count: Expected number of results (None = get all available)
            timeout: Timeout per result get operation
            
        Returns:
            List of inference results
        """
        results = []
        start_time = time.time()
        
        while True:
            # Check if we have enough results
            if expected_count and len(results) >= expected_count:
                break
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Get result
            remaining_timeout = None
            if timeout:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    break
                remaining_timeout = remaining
            
            result = self.get_result(timeout=remaining_timeout)
            if result:
                results.append(result)
            elif expected_count is None:
                # No more results available
                break
        
        return results
    
    def cancel(self, force: bool = False, timeout: float = 5.0) -> bool:
        """Cancel all pending and running tasks.
        
        Args:
            force: Whether to force terminate worker threads
            timeout: Timeout for graceful shutdown
            
        Returns:
            True if cancellation succeeded
        """
        logger.info("Cancelling inference thread pool")
        
        # Signal cancellation
        self.cancel_event.set()
        
        # Clear pending tasks
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                if task:
                    self.result_queue.put(InferenceResult(
                        task_id=task.task_id,
                        frame=None,
                        success=False,
                        error="Cancelled"
                    ))
                self.task_queue.task_done()
            except queue.Empty:
                break
        
        if force:
            # Force terminate workers
            logger.warning("Force terminating workers")
            for worker in self.workers:
                worker.force_terminate()
            return True
        else:
            # Wait for workers to finish current tasks
            logger.info(f"Waiting {timeout}s for workers to finish")
            for worker in self.workers:
                worker.join(timeout=timeout / len(self.workers))
            
            # Check if all workers stopped
            alive_workers = [w for w in self.workers if w.is_alive()]
            if alive_workers:
                logger.warning(f"{len(alive_workers)} workers still alive, forcing termination")
                for worker in alive_workers:
                    worker.force_terminate()
            
            return len(alive_workers) == 0
    
    def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        """Shutdown the thread pool.
        
        Args:
            wait: Whether to wait for pending tasks
            timeout: Timeout for waiting
        """
        if self._shutdown:
            return
        
        logger.info("Shutting down inference thread pool")
        
        if not wait:
            # Cancel all tasks
            self.cancel(force=False, timeout=timeout)
        
        # Send poison pills to workers
        for _ in self.workers:
            try:
                self.task_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout / len(self.workers))
        
        # Force terminate any remaining workers
        for worker in self.workers:
            if worker.is_alive():
                worker.force_terminate()
        
        self.workers.clear()
        self._started = False
        self._shutdown = True
        self.cancel_event.clear()
        
        logger.info("Inference thread pool shutdown complete")
    
    def is_alive(self) -> bool:
        """Check if any worker is still running."""
        return any(w.is_alive() for w in self.workers)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Dict with pool statistics
        """
        return {
            "num_workers": self.num_workers,
            "active_workers": sum(1 for w in self.workers if w.is_alive()),
            "pending_tasks": self.task_queue.qsize(),
            "completed_results": self.result_queue.qsize(),
            "started": self._started,
            "shutdown": self._shutdown,
            "cancelled": self.cancel_event.is_set(),
        }
