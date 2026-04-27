# 修复 FP16 推理时 dtype 不匹配错误

**时间:** 2026-04-22 07:44:00  
**状态:** ✅ 已解决

---

## 问题描述

使用 VFI 模型进行视频帧插值时，当启用 `fp16=True` 进行半精度推理时，出现以下错误：

```
RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same
```

错误发生在 `core/processor.py:217`，表明模型权重是 `float16` (Half)，但输入张量仍然是 `float32`。

---

## 根因分析

### 问题定位

1. **错误来源:** `core/torch_backend/vfi_torch/rife/__init__.py`

2. **代码逻辑:**
   - 当 `fp16=True` 时，模型权重在第 358-359 行被转换为半精度：
     ```python
     if self._config.fp16 and self.device.type == "cuda":
         self._model = self._model.half()
     ```
   - 输入帧在第 396-398 行被转换为半精度：
     ```python
     if self._config.fp16 and self.device.type == "cuda":
         frame0 = frame0.half()
         frame1 = frame1.half()
     ```

3. **问题代码 (第 230-231 行):**
   ```python
   if not torch.is_tensor(timestep):
       timestep = torch.ones(img0.shape[0], 1, img0.shape[2], img0.shape[3], device=img0.device) * timestep
   ```
   
   `timestep` 张量使用默认的 `float32` dtype 创建，没有匹配输入帧的 dtype。

### 参考 ComfyUI-Frame-Interpolation

对比 ComfyUI-Frame-Interpolation 的 `vfi_utils.py`，发现其在 `_generic_frame_loop` 函数中正确处理了 dtype：

```python
# Line 157
dtype=torch.float16,

# Line 206-210
middle_frame = return_middle_frame_function(
    frame0.to(DEVICE),
    frame1.to(DEVICE),
    timestep,
    *return_middle_frame_function_args
).detach().cpu().to(dtype=dtype)
```

关键区别：ComfyUI 的实现在数据流转过程中始终保持 dtype 一致性。

---

## 解决方案

### 修复代码

**文件:** `core/torch_backend/vfi_torch/rife/__init__.py`  
**位置:** 第 230-231 行

**修改前:**
```python
if not torch.is_tensor(timestep):
    timestep = torch.ones(img0.shape[0], 1, img0.shape[2], img0.shape[3], device=img0.device) * timestep
```

**修改后:**
```python
if not torch.is_tensor(timestep):
    timestep = torch.ones(img0.shape[0], 1, img0.shape[2], img0.shape[3], device=img0.device, dtype=img0.dtype) * timestep
```

### 修改说明

添加 `dtype=img0.dtype` 参数，确保 `timestep` 张量的 dtype 与输入帧保持一致。这样当输入帧是 `float16` 时，timestep 也会自动转换为 `float16`。

---

## 验证

修复后，FP16 推理应该能正常工作，不再出现 dtype 不匹配的错误。

---

## 相关文件

- `core/torch_backend/vfi_torch/rife/__init__.py` - RIFE 模型实现
- `core/torch_backend/vfi_torch/base.py` - VFI 模型基类
- `core/torch_backend/frame_processor.py` - 帧处理逻辑
- `ComfyUI-Frame-Interpolation/vfi_utils.py` - 参考实现
