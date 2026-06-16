# Event System

VFI-gui uses **Blinker** signals for Qt-independent event communication between core and UI layers.

## Why Blinker?

Previously, `ModelManager` inherited from `QObject` and used `pyqtSignal`, creating a Qt dependency in the core layer. Blinker provides:

- **Qt Independence**: Core layer no longer depends on PyQt6
- **Thread Safety**: Safe for use across threads
- **Familiar API**: Similar to Qt signals (`connect`/`send` vs `connect`/`emit`)
- **Sender Filtering**: Connect to signals from specific senders only

## Available Signals

### Model Events

| Signal | Description | Sender |
|--------|-------------|--------|
| `engines_updated` | TensorRT engines rescanned | ModelManager |
| `models_updated` | Model checkpoints rescanned | ModelManager |

### Processing Events

| Signal | Description | Args |
|--------|-------------|------|
| `processing_state_changed` | Processing state changed | `state: str` |
| `processing_progress` | Progress during processing | `frame: int, total: int, fps: float` |
| `processing_finished` | Processing completed | `success: bool, message: str` |

### Queue Events

| Signal | Description | Args |
|--------|-------------|------|
| `queue_changed` | Queue contents changed | None |
| `queue_item_status_changed` | Item status changed | `index: int, status: QueueItemStatus` |

### Download Events

| Signal | Description | Args |
|--------|-------------|------|
| `download_progress` | Download progress | `progress: int, message: str` |
| `download_finished` | Download completed | `success: bool, message: str` |

## Usage Examples

### Subscribing to Events

```python
from core.events import models_updated, engines_updated

# Basic subscription
def on_models_updated(sender):
    print(f"Models updated by {sender}")

models_updated.connect(on_models_updated)

# Subscribe to specific sender only
model_manager = ModelManager(config)
models_updated.connect(on_models_updated, sender=model_manager)
```

### Emitting Events

```python
from core.events import models_updated

class ModelManager:
    def _scan_checkpoints(self):
        # ... scan logic ...
        models_updated.send(self)
```

### Disconnecting

```python
# Disconnect specific callback
models_updated.disconnect(on_models_updated)

# Disconnect all callbacks for a signal
models_updated.receivers.clear()
```

### Qt Integration

For UI components that need Qt signals, create a bridge:

```python
from PyQt6.QtCore import QObject, pyqtSignal
from core.events import models_updated

class ModelBridge(QObject):
    models_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        models_updated.connect(self._on_models_updated)

    def _on_models_updated(self, sender):
        self.models_changed.emit()

# In UI code
bridge = ModelBridge()
bridge.models_changed.connect(self.refresh_ui)
```

## Migration from pyqtSignal

### Before (Qt-dependent)

```python
from PyQt6.QtCore import QObject, pyqtSignal

class ModelManager(QObject):
    models_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def refresh(self):
        # ...
        self.models_updated.emit()

# Usage
manager = ModelManager()
manager.models_updated.connect(callback)
```

### After (Qt-independent)

```python
from core.events import models_updated

class ModelManager:
    def __init__(self):
        pass  # No parent parameter needed

    def refresh(self):
        # ...
        models_updated.send(self)

# Usage
manager = ModelManager()
models_updated.connect(callback, sender=manager)
```

## Thread Safety

Blinker signals are thread-safe. Signals emitted from background threads will call subscribers in the emitter's thread. For UI updates from background threads, use Qt's signal/slot mechanism for thread-safe UI updates:

```python
from core.events import processing_progress
from PyQt6.QtCore import QMetaObject, Qt

def on_progress(sender, frame, total, fps):
    # This runs in the processing thread
    # Use Qt's queued connection for thread-safe UI update
    QMetaObject.invokeMethod(
        self.progress_bar,
        "setValue",
        Qt.QueuedConnection,
        Q_ARG(int, int(frame / total * 100))
    )

processing_progress.connect(on_progress)
```

## See Also

- [Blinker Documentation](https://blinker.readthedocs.io/)
- [core/events.py](../core/events.py) - Signal definitions
- [core/models.py](../core/models.py) - Example usage in ModelManager