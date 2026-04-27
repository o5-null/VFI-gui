"""Frame processing loop with memory management.

This module provides the core frame processing loop for video frame interpolation,
handling:
- Frame pair iteration
- Memory management and cache clearing
- Batch processing for improved throughput
- Progress callbacks
"""

import gc
import time
import typing
from typing import Callable, List, Optional, Tuple, Union

import torch

from .base import (
    VFIBaseModel,
    InterpolationConfig,
    InterpolationResult,
    InterpolationStateList,
    DType,
    DTYPE_MAP,
    clear_cuda_cache,
    assert_batch_size,
    preprocess_frames,
    postprocess_frames,
)


class FrameProcessor:
    """Process frames through a VFI model.
    
    Handles the frame processing loop with memory management, batching,
    and progress callbacks.
    
    Usage:
        processor = FrameProcessor(model)
        result = processor.process(
            frames,
            multiplier=2,
            clear_cache_after_n_frames=10,
        )
    """
    
    def __init__(
        self,
        model: VFIBaseModel,
        config: Optional[InterpolationConfig] = None,
    ):
        """
        Args:
            model: The VFI model to use for interpolation.
            config: Processing configuration. Uses defaults if None.
        """
        self.model = model
        self.config = config or InterpolationConfig()
    
    def process(
        self,
        frames: torch.Tensor,
        multiplier: Union[int, List[int]] = 2,
        clear_cache_after_n_frames: int = 10,
        interpolation_states: Optional[InterpolationStateList] = None,
        batch_size: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> InterpolationResult:
        """Process frames through the VFI model.
        
        Args:
            frames: Input frames in NHWC format [N, H, W, C].
            multiplier: Interpolation multiplier. Can be int or list of ints.
            clear_cache_after_n_frames: Clear CUDA cache after this many frames.
            interpolation_states: Optional frame skip configuration.
            batch_size: Number of frame pairs to process per batch.
            progress_callback: Optional callback(current, total) for progress.
            
        Returns:
            InterpolationResult with output frames and metadata.
        """
        start_time = time.time()
        
        # Preprocess frames: NHWC -> NCHW
        frames = preprocess_frames(frames)
        original_count = len(frames)
        
        # Validate
        assert_batch_size(frames, min_frames=2, model_name=self.model.MODEL_TYPE)
        
        # Normalize multiplier to list
        n_pairs = len(frames) - 1
        if isinstance(multiplier, int):
            multipliers = [multiplier] * n_pairs
        else:
            multipliers = list(multiplier)
            multipliers += [2] * (n_pairs - len(multipliers))
        
        # Process frames
        output_frames = self._process_frame_loop(
            frames,
            multipliers,
            clear_cache_after_n_frames,
            interpolation_states,
            batch_size,
            progress_callback,
        )
        
        processing_time = time.time() - start_time
        output_count = len(output_frames)
        
        # Postprocess: NCHW -> NHWC
        output_frames = postprocess_frames(output_frames)
        
        return InterpolationResult(
            frames=output_frames,
            original_count=original_count,
            output_count=output_count,
            processing_time=processing_time,
        )
    
    def _process_frame_loop(
        self,
        frames: torch.Tensor,
        multipliers: List[int],
        clear_cache_after_n_frames: int,
        interpolation_states: Optional[InterpolationStateList],
        batch_size: int,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> torch.Tensor:
        """Core frame processing loop.
        
        Args:
            frames: Preprocessed frames [N, C, H, W].
            multipliers: Per-pair interpolation multipliers.
            clear_cache_after_n_frames: Cache clearing interval.
            interpolation_states: Frame skip configuration.
            batch_size: Batch size for processing.
            progress_callback: Progress callback.
            
        Returns:
            Output frames tensor.
        """
        dtype = self.model.torch_dtype
        device = self.model.device
        
        # Calculate total output frames
        total_output = sum(multipliers) + 1
        
        # Pre-allocate output tensor
        output_frames = torch.zeros(
            total_output, *frames.shape[1:],
            dtype=dtype, device="cpu"
        )
        out_idx = 0
        
        # Build task list: (pair_idx, timestep)
        tasks: List[Tuple[int, float]] = []
        tasks_per_pair: Dict[int, int] = {}
        
        for pair_idx in range(len(frames) - 1):
            # Check if pair should be skipped
            if interpolation_states is not None and interpolation_states.is_frame_skipped(pair_idx):
                tasks_per_pair[pair_idx] = 0
                continue
            
            m = multipliers[pair_idx]
            n_steps = max(m - 1, 0)
            tasks_per_pair[pair_idx] = n_steps
            
            for step in range(1, m):
                tasks.append((pair_idx, step / m))
        
        # Storage for intermediate frames
        results: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(frames) - 1)}
        
        # Process tasks
        frames_since_cache_clear = 0
        pos = 0
        
        with torch.inference_mode():
            while pos < len(tasks):
                # Get batch of tasks
                batch_tasks = tasks[pos:pos + batch_size]
                
                # Prepare batch inputs
                frame0_list, frame1_list, timestep_list = [], [], []
                for pair_idx, dt in batch_tasks:
                    frame0_list.append(frames[pair_idx:pair_idx + 1])
                    frame1_list.append(frames[pair_idx + 1:pair_idx + 2])
                    timestep_list.append(dt)
                
                # Stack and move to device
                frame0_batch = torch.cat(frame0_list, dim=0).to(device, dtype=dtype)
                frame1_batch = torch.cat(frame1_list, dim=0).to(device, dtype=dtype)
                
                # Process batch
                for idx, (pair_idx, dt) in enumerate(batch_tasks):
                    # Interpolate single frame
                    middle_frame = self.model.interpolate(
                        frame0_batch[idx:idx + 1],
                        frame1_batch[idx:idx + 1],
                        dt,
                    )
                    
                    # Store result
                    results[pair_idx].append(
                        middle_frame.clamp(0, 1).detach().cpu().to(dtype=dtype)
                    )
                    tasks_per_pair[pair_idx] -= 1
                    
                    # Check if pair is complete
                    if tasks_per_pair[pair_idx] == 0:
                        frames_since_cache_clear += 1
                        
                        # Clear cache periodically
                        if frames_since_cache_clear >= clear_cache_after_n_frames:
                            print("[FrameProcessor] Clearing cache...")
                            clear_cuda_cache()
                            frames_since_cache_clear = 0
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(out_idx, total_output)
                
                pos += len(batch_tasks)
        
        # Assemble output: each original frame + its interpolated frames
        output_list: List[torch.Tensor] = []
        for pair_idx in range(len(frames) - 1):
            output_list.append(frames[pair_idx:pair_idx + 1].to(dtype=dtype))
            for mid in results[pair_idx]:
                output_list.append(mid)
        output_list.append(frames[-1:].to(dtype=dtype))
        
        # Concatenate all frames
        output_frames = torch.cat(output_list, dim=0)
        
        # Final cache clear
        print("[FrameProcessor] Final cache clear...")
        clear_cuda_cache()
        
        print(f"[FrameProcessor] Done! {len(output_frames)} frames generated")
        
        return output_frames
    
    def process_with_timestep(
        self,
        frames: torch.Tensor,
        multiplier: int = 2,
        clear_cache_after_n_frames: int = 10,
        interpolation_states: Optional[InterpolationStateList] = None,
        batch_size: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> InterpolationResult:
        """Process frames using timestep-based interpolation.
        
        This method uses explicit timesteps for each interpolated frame,
        which is more efficient for models that support it.
        
        Args:
            frames: Input frames in NHWC format.
            multiplier: Interpolation multiplier.
            clear_cache_after_n_frames: Cache clearing interval.
            interpolation_states: Frame skip configuration.
            batch_size: Batch size for processing.
            progress_callback: Progress callback.
            
        Returns:
            InterpolationResult with output frames.
        """
        return self.process(
            frames,
            multiplier,
            clear_cache_after_n_frames,
            interpolation_states,
            batch_size,
            progress_callback,
        )
    
    def process_recursive(
        self,
        frames: torch.Tensor,
        multiplier: int = 2,
        clear_cache_after_n_frames: int = 10,
        interpolation_states: Optional[InterpolationStateList] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> InterpolationResult:
        """Process frames using recursive interpolation.
        
        This method recursively interpolates frames, which is useful for
        models that don't support explicit timesteps.
        
        Args:
            frames: Input frames in NHWC format.
            multiplier: Interpolation multiplier.
            clear_cache_after_n_frames: Cache clearing interval.
            interpolation_states: Frame skip configuration.
            progress_callback: Progress callback.
            
        Returns:
            InterpolationResult with output frames.
        """
        start_time = time.time()
        
        # Preprocess frames
        frames = preprocess_frames(frames)
        original_count = len(frames)
        
        assert_batch_size(frames, min_frames=2, model_name=self.model.MODEL_TYPE)
        
        dtype = self.model.torch_dtype
        device = self.model.device
        
        def recursive_inference(
            frame0: torch.Tensor,
            frame1: torch.Tensor,
            n: int,
        ) -> List[torch.Tensor]:
            """Recursively interpolate between two frames."""
            if n <= 0:
                return []
            
            # Get middle frame
            middle = self.model.interpolate(
                frame0.to(device, dtype=dtype),
                frame1.to(device, dtype=dtype),
                0.5,
            ).detach().cpu().to(dtype=dtype)
            
            if n == 1:
                return [middle]
            
            # Recursively interpolate each half
            first_half = recursive_inference(frame0, middle, n // 2)
            second_half = recursive_inference(middle, frame1, n // 2)
            
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]
        
        output_list: List[torch.Tensor] = []
        frames_since_cache_clear = 0
        
        for frame_idx in range(len(frames) - 1):
            # Check if pair should be skipped
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_idx):
                output_list.append(frames[frame_idx:frame_idx + 1].to(dtype=dtype))
                continue
            
            # Add first frame
            output_list.append(frames[frame_idx:frame_idx + 1].to(dtype=dtype))
            
            # Get interpolated frames
            middle_frames = recursive_inference(
                frames[frame_idx:frame_idx + 1],
                frames[frame_idx + 1:frame_idx + 2],
                multiplier - 1,
            )
            output_list.extend(middle_frames)
            
            # Cache management
            frames_since_cache_clear += 1
            if frames_since_cache_clear >= clear_cache_after_n_frames:
                print("[FrameProcessor] Clearing cache...")
                clear_cuda_cache()
                frames_since_cache_clear = 0
            
            if progress_callback:
                progress_callback(len(output_list), original_count * multiplier)
        
        # Add final frame
        output_list.append(frames[-1:].to(dtype=dtype))
        
        processing_time = time.time() - start_time
        output_frames = torch.cat(output_list, dim=0)
        
        # Final cache clear
        clear_cuda_cache()
        
        # Postprocess
        output_frames = postprocess_frames(output_frames)
        
        return InterpolationResult(
            frames=output_frames,
            original_count=original_count,
            output_count=len(output_frames),
            processing_time=processing_time,
        )


# Type alias for dict
Dict = dict
