"""Dark theme stylesheet for VFI-gui application."""

DARK_THEME = """
/* Global Settings */
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 10pt;
}

QMainWindow {
    background-color: #1e1e1e;
}

/* Menu Bar */
QMenuBar {
    background-color: #2d2d2d;
    border-bottom: 1px solid #3d3d3d;
    padding: 2px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 5px 10px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #3d3d3d;
}

QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}

QMenu::item {
    padding: 5px 30px 5px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #0078d4;
}

QMenu::separator {
    height: 1px;
    background-color: #3d3d3d;
    margin: 5px 10px;
}

/* Tool Bar */
QToolBar {
    background-color: #2d2d2d;
    border-bottom: 1px solid #3d3d3d;
    padding: 5px;
    spacing: 5px;
}

QToolBar::separator {
    width: 1px;
    background-color: #3d3d3d;
    margin: 5px;
}

/* Buttons */
QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #4d4d4d;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #4d4d4d;
    border-color: #5d5d5d;
}

QPushButton:pressed {
    background-color: #2d2d2d;
}

QPushButton:disabled {
    background-color: #2d2d2d;
    color: #5d5d5d;
    border-color: #3d3d3d;
}

QPushButton#primaryButton {
    background-color: #0078d4;
    border-color: #0078d4;
}

QPushButton#primaryButton:hover {
    background-color: #1084d8;
}

QPushButton#primaryButton:pressed {
    background-color: #006cbd;
}

QPushButton#dangerButton {
    background-color: #d42a2a;
    border-color: #d42a2a;
}

QPushButton#dangerButton:hover {
    background-color: #e03535;
}

/* Input Fields */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: #0078d4;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #0078d4;
}

QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
    background-color: #252525;
    color: #5d5d5d;
}

/* ComboBox */
QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px 8px;
    min-width: 100px;
}

QComboBox:hover {
    border-color: #4d4d4d;
}

QComboBox:focus {
    border-color: #0078d4;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #e0e0e0;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    selection-background-color: #0078d4;
    outline: none;
}

/* SpinBox */
QSpinBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px 8px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #0078d4;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #3d3d3d;
    border: none;
    width: 20px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #4d4d4d;
}

/* CheckBox */
QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #4d4d4d;
    border-radius: 3px;
    background-color: #2d2d2d;
}

QCheckBox::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

QCheckBox::indicator:hover {
    border-color: #5d5d5d;
}

/* RadioButton */
QRadioButton {
    spacing: 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #4d4d4d;
    border-radius: 9px;
    background-color: #2d2d2d;
}

QRadioButton::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

/* GroupBox */
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #0078d4;
}

/* ScrollBar */
QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #4d4d4d;
    border-radius: 6px;
    min-height: 30px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5d5d5d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 12px;
    border: none;
}

QScrollBar::handle:horizontal {
    background-color: #4d4d4d;
    border-radius: 6px;
    min-width: 30px;
    margin: 2px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #5d5d5d;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* Slider */
QSlider::groove:horizontal {
    background-color: #3d3d3d;
    height: 4px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background-color: #0078d4;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #1084d8;
}

/* ProgressBar */
QProgressBar {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #0078d4;
    border-radius: 3px;
}

/* TabWidget */
QTabWidget::pane {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    background-color: #1e1e1e;
}

QTabBar::tab {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 16px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    border-color: #0078d4;
    border-bottom: 2px solid #0078d4;
}

QTabBar::tab:hover:!selected {
    background-color: #3d3d3d;
}

/* ListWidget */
QListWidget {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    outline: none;
}

QListWidget::item {
    padding: 5px;
    border-radius: 2px;
}

QListWidget::item:selected {
    background-color: #0078d4;
}

QListWidget::item:hover:!selected {
    background-color: #3d3d3d;
}

/* TreeWidget */
QTreeWidget {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    outline: none;
}

QTreeWidget::item {
    padding: 3px;
}

QTreeWidget::item:selected {
    background-color: #0078d4;
}

QTreeWidget::item:hover:!selected {
    background-color: #3d3d3d;
}

/* Splitter */
QSplitter::handle {
    background-color: #3d3d3d;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* ScrollArea */
QScrollArea {
    border: none;
    background-color: transparent;
}

/* Label */
QLabel {
    background-color: transparent;
}

/* Tooltip */
QToolTip {
    background-color: #2d2d2d;
    border: 1px solid #4d4d4d;
    border-radius: 4px;
    padding: 5px;
    color: #e0e0e0;
}

/* Status Bar */
QStatusBar {
    background-color: #2d2d2d;
    border-top: 1px solid #3d3d3d;
}

/* Dock Widget */
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background-color: #2d2d2d;
    padding: 6px;
}

/* Frame for sections */
QFrame#sectionFrame {
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    background-color: #252525;
}

/* Drop zone */
QFrame#dropZone {
    border: 2px dashed #4d4d4d;
    border-radius: 8px;
    background-color: #252525;
}

QFrame#dropZone:hover {
    border-color: #0078d4;
    background-color: #2d2d2d;
}
"""
