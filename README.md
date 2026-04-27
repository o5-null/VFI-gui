# VFI-gui

<p align="center">
  <strong>Video Frame Interpolation GUI</strong><br>
  A PyQt6 desktop application for AI-powered video processing
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#supported-models">Supported Models</a> •
  <a href="#development">Development</a>
</p>

---

## Features

- **Video Frame Interpolation** - RIFE-based frame interpolation for smooth slow-motion effects
- **Video Upscaling** - ESRGAN/Real-CUGAN super-resolution upscaling
- **Scene Detection** - Automatic scene change detection for optimal interpolation
- **Batch Processing** - Queue multiple videos for automated processing
- **Multi-language Support** - English, Simplified Chinese, Traditional Chinese
- **Multi-GPU Support** - NVIDIA CUDA and Intel XPU acceleration
- **Dark Theme** - Modern dark UI design

## Screenshots

*Coming soon*

## Requirements

- **Python**: 3.12+
- **OS**: Windows 10/11 (primary), Linux (experimental)
- **GPU** (optional but recommended):
  - NVIDIA GPU with CUDA support
  - Intel GPU with XPU support
  - CPU fallback available

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/o5-null/VFI-gui.git
cd VFI-gui

# Create virtual environment
uv venv --python 3.12

# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
uv pip install -r requirements.txt

# Run the application
python main.py
```

### VapourSynth Installation

VapourSynth R74+ is required for video processing:

```bash
# R74+ (recommended)
uv pip install vapoursynth

# Or use the portable version included in the plugin directory
```

### GPU Runtime Setup

For GPU acceleration, install the appropriate PyTorch version:

**NVIDIA CUDA:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**Intel XPU:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

## Usage

### Basic Workflow

1. **Open Video** - Click "Open" or use `Ctrl+O` to select a video file
2. **Configure Pipeline** - Adjust interpolation, upscaling, and output settings
3. **Start Processing** - Click "Start" to begin processing
4. **Monitor Progress** - View real-time progress and logs
5. **Check Output** - Processed video saves to the output directory

### Batch Processing

1. Add videos to queue using "Add to Queue" button
2. Or use "Open Folder" to add all videos from a directory
3. Configure settings for each video (optional)
4. Click "Start Batch" to process all queued videos

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open video file |
| `Ctrl+Shift+O` | Open folder |
| `Ctrl+S` | Save settings |
| `Ctrl+Shift+A` | Add to queue |
| `Ctrl+Q` | Quit application |

## Supported Models

### Frame Interpolation (RIFE)

| Model | Description |
|-------|-------------|
| RIFE 4.0 | Standard interpolation model |
| RIFE 4.3 | Improved quality |
| RIFE 4.6 | Latest stable version |
| RIFE 4.22 | High-quality interpolation |
| RIFE 4.24 | Latest model |
| RIFE 4.25 | Latest model |

### Upscaling (ESRGAN/Real-CUGAN)

- ESRGAN - General-purpose upscaling
- Real-CUGAN - Optimized for anime content

### Scene Detection

- Automatic scene change detection for optimal frame interpolation
- Configurable threshold settings

## Project Structure

```
VFI-gui/
├── core/                    # Core logic
│   ├── config.py            # Configuration management
│   ├── pipeline.py          # Processing pipeline
│   ├── processor.py         # Video processor
│   ├── runtime_manager.py   # GPU runtime detection
│   ├── torch_backend/       # PyTorch inference backend
│   │   └── vfi_torch/       # VFI model implementations
│   │       ├── rife/        # RIFE models
│   │       ├── film/        # FILM models
│   │       ├── amt/         # AMT models
│   │       └── ifrnet/      # IFRNet models
│   └── vsgan/               # VapourSynth integration
├── ui/                      # User interface
│   ├── main_window.py       # Main application window
│   ├── widgets/             # UI components
│   ├── controllers/         # UI controllers
│   └── styles/              # Stylesheets
├── locales/                 # Translation files
│   ├── zh_CN/               # Simplified Chinese
│   └── zh_TW/               # Traditional Chinese
├── config/                  # Default configuration files
├── models/                  # Model storage (gitignored)
├── output/                  # Output directory (gitignored)
└── main.py                  # Application entry point
```

## Configuration

Configuration is stored in `~/.config/vfi-gui/`:

```
~/.config/vfi-gui/
├── settings.json            # User preferences
├── pipeline.json            # Pipeline configuration
└── runtime.json             # Runtime selection
```

### Pipeline Configuration

```json
{
  "interpolation": {
    "enabled": true,
    "model": "4.22",
    "multi": 2
  },
  "upscaling": {
    "enabled": false,
    "engine": "esrgan"
  },
  "scene_detection": {
    "enabled": true,
    "threshold": 0.93
  },
  "output": {
    "codec": "hevc_nvenc",
    "quality": 22
  }
}
```

## Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/o5-null/VFI-gui.git
cd VFI-gui

# Create venv with uv
uv venv --python 3.12
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# For type checking
uv pip install pyright
```

### Code Style

- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type annotations for function parameters
- Follow snake_case naming convention

### Internationalization (i18n)

The project uses gettext for translations:

```bash
# Compile translations after editing .po files
python compile_translations.py
```

See [docs/I18N.md](docs/I18N.md) for details.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyQt6 | >=6.5.0 | GUI framework |
| torch | >=2.0.0 | Deep learning backend |
| torchvision | >=0.15.0 | Image processing |
| vapoursynth | R74+ | Video processing framework |
| loguru | >=0.7.0 | Logging |
| einops | >=0.7.0 | Tensor operations |
| opencv-python | >=4.8.0 | Image/video processing |
| numpy | >=1.24.0 | Numerical computing |

## Troubleshooting

### "Couldn't detect vapoursynth installation path"

Install VapourSynth R74+:
```bash
uv pip install vapoursynth
```

### GPU Not Detected

1. Ensure you have the correct PyTorch version installed
2. Check CUDA/XPU drivers are up to date
3. Verify GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`

### Import Errors

The application uses runtime environments. Ensure:
- `runtime/cuda/` or `runtime/xpu/` directories exist for GPU acceleration
- Or the application will fall back to CPU mode

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - TensorRT video processing
- [RIFE](https://github.com/hzwer/RIFE) - Real-Time Intermediate Flow Estimation
- [ESRGAN](https://github.com/xinntao/ESRGAN) - Enhanced Super-Resolution GAN
- [VapourSynth](https://github.com/vapoursynth/vapoursynth) - Video processing framework
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework

## Related Projects

- [VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - The backend TensorRT framework
- [RIFE](https://github.com/hzwer/RIFE) - Frame interpolation model
- [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) - VapourSynth ML runtime

---

<p align="center">
  Made with ❤️ for video enthusiasts
</p>
