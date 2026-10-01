# ONNX 模型资产扩充：RIFE 全版本 + IFRNet/CAIN/FILM/FLAVR

- 日期：2026-10-01 19:00
- 范围：`VFI-gui`（参考仓库只读）
- 前置：`docs/dev-docs/20261001-183500-rife-onnx-all-models-smoke-test.md`（两后端 6 版本冒烟）

## 任务概述

在完成 ONNX / TensorRT-RTX 两后端 RIFE 冒烟后，继续：

1. 拉取更多 RIFE 版本（含 lite / lite_ensemble / heavy / ensemble 变体）
2. 拉取其他模型已公开的 ONNX（IFRNet、CAIN）
3. 对无 ONNX 但易导出的模型（FILM、FLAVR）转换一份 ONNX

**范围约束（用户决策）**：不扩展推理后端，仅准备 ONNX 文件；新模型作为资产就位，后端契约（当前仅 RIFE 可被加载）暂不变。

## 资产落盘结果

### RIFE（`models/rife_v2/`，41 个 7ch ONNX）

来源：vs-mlrt `external-models` release 的独立包 `rife_v4.{7..26}.7z` + 大包 `rife_v2_v4.7z`

| 类别 | 数量 | 版本 |
|------|------|------|
| 基础 | 25 | 4.0、4.2–4.15、4.17–4.26 |
| lite | 8 | 4.12–4.17、4.22、4.25 |
| lite_ensemble | 6 | 4.12–4.17 |
| heavy | 2 | 4.25、4.26 |

> 4.1 / 4.16 上游仅有 lite 变体；4.0 / 4.6 仅存在于大包 `rife_v2_v4.7z`。

### RIFE ensemble（`models/rife_ensemble/`，6 个 11ch ONNX）

来源：`rife_ensemble_v1.7z` 的 `rife\` 目录（11 通道输入）。

版本：4.0、4.2、4.3、4.4、4.5、4.6

### 其他模型（上游公开 ONNX）

| 模型 | 文件 | 大小 | 来源 |
|------|------|------|------|
| IFRNet | `models/ifrnet/ifrnet-gopro.onnx` | 20,096,394 B | `NoonLicht/MoonApp-IFRNet-ONNX` |
| IFRNet | `models/ifrnet/ifrnet-vimeo.onnx` | 20,096,394 B | 同上 |
| CAIN | `models/cain/cain.onnx` | 172,013,750 B | `NoonLicht/MoonApp-CAIN-ONNX` release v1 |

### 自导出 ONNX（FILM / FLAVR）

| 模型 | 文件 | 大小 | 导出脚本 |
|------|------|------|----------|
| FILM | `models/film/film_net_fp32.onnx` | 138,102,763 B | `scripts/export_film_onnx.py` |
| FLAVR 2x | `models/flavr/FLAVR_2x.onnx` | 168,283,706 B | `scripts/export_flavr_onnx.py` |
| FLAVR 4x | `models/flavr/FLAVR_4x.onnx` | 168,376,334 B | 同上 |
| FLAVR 8x | `models/flavr/FLAVR_8x.onnx` | 168,527,520 B | 同上 |

## 校验结果

新增可复用校验脚本 `scripts/validate_onnx_assets.py`：

- 文件名含 `rife` 的走**恒等检查**（frame0==frame1 时输出 MAE < 0.02）
- 其他模型仅做**结构校验**（onnxruntime 加载 + 输入输出名/shape/dtype/finite）
- 11ch 通道布局直接复用项目 `core/backends/rife_input.py:pack_rife_input`，保证与后端约定零漂移

| 目录 | 通过/总数 | 说明 |
|------|-----------|------|
| `models/rife_v2` | **41/41** | identity MAE 0.0016–0.0189 |
| `models/rife_ensemble` | **3/6** | 4.0=0.0089、4.5=0.0085、4.6=0.0085 通过；4.2=0.0340、4.4=0.0615、4.3=0.0809 超阈值 |
| `models/ifrnet` | **2/2** | in: img0/img1[1,3,h,w] + timestep[1] → frame[1,3,h,w] |
| `models/cain` | **1/1** | in: input[1,6,h,w] → frame[N,3,h,w] |
| `models/film` | **1/1** | in: img0/img1[1,3,256,256] + timestep[1] → output[1,3,256,256] |
| `models/flavr` | **3/3** | in: frames[1,12,256,256] → output[1,3/9/21,256,256] |

> 11ch ensemble 的 3 个超阈值项：模型结构本身在恒等输入下即有残余误差（非布局错误——布局修正后已从 0.27 降到 0.03–0.08）。该项为信息记录，不影响 7ch 资产。

## I/O 契约（供后端集成参考）

- **IFI 通用（IFRNet/FILM）**：`img0[1,3,H,W]` + `img1[1,3,H,W]` + `timestep[1]` → `frame[1,3,H,W]`，RGB [0,1]
- **CAIN**：`input[1,6,H,W]`（两帧通道拼接）→ `frame[N,3,H,W]`
- **FLAVR**：`frames[1,12,H,W]`（4 帧×3ch 拼接，RGB [0,1]）→ `output[1,3*n,H,W]`（n=1/3/7 对应 2x/4x/8x），静态 256×256
- **RIFE 7ch**：`[1,7,H,W]`（f0(3)+f1(3)+t(1)）；**RIFE 11ch**：追加 horizontal/vertical/multiplier_h/multiplier_w 四平面

## 导出注意事项

### FILM

- 本地 `models/film/film_net_fp32.pt` 为 TorchScript，其内部**把时间硬编码为 0.5**（`torch.full_like(batch_dt, .5)`），原模型忽略 timestep。
- TorchScript 直导 ONNX 失败于动态 `interpolate.size`（`size` 来自动态 `aten::slice(aten::size)`）；monkeypatch symbolic 会产生非法 ONNX。
- 方案：用纯 PyTorch **忠实重建 FILM 架构** + 严格加载 TorchScript `state_dict`（82 参，零 missing/unexpected）→ trace 导出；并把 `batch_dt` 真正接入光流缩放。
- 结果：t=0.5 时与原模型逐位一致（max abs diff 0.0）；ONNX@t=0.5 MAE 4.2e-06；t=0.25/0.75 与 TorchScript 差异 ≈0.17（设计差异，原模型忽略 t）。

### FLAVR

- `UNet3D3D` 支持 4 帧输入，`n_outputs=1/3/7` 对应 2x/4x/8x。
- `resnet_3d.useBias` 是**模块级全局**，仅被 `UNet3D3D` 在 `n_outputs>1` 时翻成 `True` 且不复原，跨模型构建会污染 → 已按 checkpoint 显式设置。
- 静态 256×256、opset 17，导出后与 PyTorch 对比 MAE < 1e-5。

## 发现的隐性缺陷（未修，建议后续评估）

`core/pytorch_models/vfi_torch/flavr/resnet_3d.py`：

- encoder BatchNorm 为硬编码 `nn.BatchNorm3d`，且 `unet_18()` 会无条件重置该全局 → 直接 `UNet3D3D(...)` 会多出 100 个 BN key。
- 权重文件**完全不含 encoder BN 张量**，而 `FLAVRModel.load_model` 用 `strict=False` 静默吞掉 → BN 停留默认值（≈恒等）。
- 推理"看起来正常"，但属隐性不一致。参考实现 `ComfyUI-Frame-Interpolation/vfi_models/flavr/flavr_arch.py:149` 为 `block(pretrained=False, bn=batchnorm)`（无 BN）。
- 导出脚本通过替换 `unet_18` 构建无 BN encoder，实现 1:1 加载。是否修正项目 `resnet_3d.py` 待决策。

## 输出文件

- `scripts/validate_onnx_assets.py`（新增，通用 ONNX 资产校验）
- `scripts/export_film_onnx.py`（新增，FILM ONNX 导出）
- `scripts/export_flavr_onnx.py`（新增，FLAVR 2x/4x/8x ONNX 导出）
- `models/rife_v2/`（41 个）、`models/rife_ensemble/`（6 个）
- `models/ifrnet/`（2 个）、`models/cain/`（1 个）
- `models/film/film_net_fp32.onnx`、`models/flavr/FLAVR_{2x,4x,8x}.onnx`

## 经验总结

1. ONNX 权重无自动下载，RIFE 全系从 vs-mlrt release 取，`_lite`/`_heavy`/`_ensemble` 为独立资产。
2. 校验 11ch RIFE 必须复用 `pack_rife_input`，手工 meshgrid 极易写错布局（会导致 MAE 从 0.008 假性放大到 0.27）。
3. FILM 官方权重用 TorchScript 硬编码 t=0.5，直导 ONNX 不可行；"重建架构 + 加载 state_dict" 是可行且忠实的路径。
4. FLAVR 存在模块级全局状态（`useBias` / `batchnorm`），批量构建多模型时须显式逐模型设置。
5. 「上游有 ONNX」与「项目后端可加载」是两回事：本次仅准备资产，未扩展 `BackendType`/`SUPPORTED_MODELS`。
