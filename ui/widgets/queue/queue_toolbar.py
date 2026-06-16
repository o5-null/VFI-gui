"""QueueToolbar — queue action toolbar.

Simple toolbar for queue operations: add/remove/clear.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy
from PyQt6.QtCore import QEvent, Qt, QSize

from ui.styles.theme import Theme
from ui.styles.icons import IconManager

if TYPE_CHECKING:
    from ui.controllers.queue_controller import QueueController


class QueueToolbar(QWidget):
    """Queue action toolbar — add/remove/clear operations.
    
    Features:
        - Add button
        - Remove button
        - Clear button
        - Horizontal layout with icons
    """

    def __init__(self, ctrl: "QueueController", parent=None):
        """Initialize QueueToolbar.
        
        Args:
            ctrl: QueueController for queue actions
            parent: Parent widget
        """
        super().__init__(parent)
        self._ctrl = ctrl
        self.setObjectName("queueToolbar")
        self._setup_ui()
        self.retranslate_ui()

    def changeEvent(self, a0: QEvent | None) -> None:
        """Handle language change for i18n."""
        if a0 is not None and a0.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(a0)

    def retranslate_ui(self) -> None:
        """Update all user-visible text for i18n."""
        if hasattr(self, "_add_btn"):
            self._add_btn.setText(self.tr("Add"))
            self._remove_btn.setText(self.tr("Remove"))
            self._clear_btn.setText(self.tr("Clear"))

    def _setup_ui(self) -> None:
        """Create widget structure."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            Theme.PADDING_SM, Theme.PADDING_SM,
            Theme.PADDING_SM, Theme.PADDING_SM
        )
        layout.setSpacing(Theme.SPACING_SM)

        # Add button
        self._add_btn = QPushButton()
        self._add_btn.setObjectName("queueToolbarAddBtn")
        add_icon = IconManager.get("add", Theme.TEXT_PRIMARY, QSize(16, 16))
        self._add_btn.setIcon(add_icon)
        self._add_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._add_btn)

        # Remove button
        self._remove_btn = QPushButton()
        self._remove_btn.setObjectName("queueToolbarRemoveBtn")
        remove_icon = IconManager.get("remove", Theme.TEXT_PRIMARY, QSize(16, 16))
        self._remove_btn.setIcon(remove_icon)
        self._remove_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._remove_btn)

        # Clear button
        self._clear_btn = QPushButton()
        self._clear_btn.setObjectName("queueToolbarClearBtn")
        clear_icon = IconManager.get("clear", Theme.TEXT_PRIMARY, QSize(16, 16))
        self._clear_btn.setIcon(clear_icon)
        self._clear_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._clear_btn.clicked.connect(self._clear_queue)
        layout.addWidget(self._clear_btn)

        layout.addStretch()

    def _clear_queue(self) -> None:
        """Clear entire queue."""
        self._ctrl.clear_queue()


__all__ = ["QueueToolbar"]