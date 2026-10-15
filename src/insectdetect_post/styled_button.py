"""Custom styled button widget for Qt applications.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Paints its own rounded background, border, and text color based on hover, pressed,
and enabled state, replacing Qt stylesheets for consistent cross-platform theming.

Classes:
    StyledButton: Button with theme-aware styling and state handling.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QEnterEvent, QMouseEvent, QPainter, QPainterPath, QPaintEvent
from PySide6.QtWidgets import QPushButton


class StyledButton(QPushButton):
    """Button with theme-aware styling and state handling."""

    _TEXT_PADDING = 6
    _BORDER_WIDTH = 1
    _BORDER_RADIUS = 4

    _COLOR_DISABLED = QColor("#BDBDBD")
    _COLOR_TEXT_ENABLED = QColor("#F5F5F5")
    _COLOR_TEXT_DISABLED = QColor("#757575")

    def __init__(self, text: str, bg_color: str, font_size: int = 9) -> None:
        """Initialize styled button with custom color and font size.

        Args:
            text: Button label text.
            bg_color: Base background color as a CSS color string (e.g. hex code).
            font_size: Font point size for the button label.

        Raises:
            ValueError: If bg_color is not a valid color string.
        """
        super().__init__(text)
        color = QColor(bg_color)
        if not color.isValid():
            raise ValueError(f"Invalid color string: {bg_color}")
        self._base_color = color

        font = self.font()
        font.setBold(True)
        font.setPointSize(font_size)
        self.setFont(font)

    def sizeHint(self) -> QSize:
        """Calculate button size based on text and padding."""
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text())
        text_h = fm.height()
        w = text_w + 2 * self._TEXT_PADDING + 2 * self._BORDER_WIDTH
        h = text_h + 2 * self._TEXT_PADDING + 2 * self._BORDER_WIDTH
        return QSize(w, h)

    def _get_state_color(self) -> QColor:
        """Return background color for the current button state."""
        if not self.isEnabled():
            return self._COLOR_DISABLED
        if self.isDown():
            return self._base_color.darker(120)
        if self.underMouse():
            return self._base_color.lighter(130)
        return self._base_color

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint button background, border, and text based on the current state."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        bg_color = self._get_state_color()
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), self._BORDER_RADIUS, self._BORDER_RADIUS)
        painter.fillPath(path, bg_color)

        pen = painter.pen()
        pen.setWidthF(self._BORDER_WIDTH)
        pen.setColor(self._COLOR_DISABLED if not self.isEnabled() else bg_color.darker(140))
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        border_inset = self._BORDER_WIDTH / 2
        border_rect = QRectF(rect).adjusted(border_inset, border_inset, -border_inset, -border_inset)
        border_path = QPainterPath()
        border_path.addRoundedRect(border_rect, self._BORDER_RADIUS, self._BORDER_RADIUS)
        painter.drawPath(border_path)

        text_rect = rect.adjusted(
            self._TEXT_PADDING, self._TEXT_PADDING, -self._TEXT_PADDING, -self._TEXT_PADDING
        )
        text_color = self._COLOR_TEXT_ENABLED if self.isEnabled() else self._COLOR_TEXT_DISABLED
        painter.setPen(text_color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()

    def enterEvent(self, event: QEnterEvent) -> None:
        """Trigger a repaint on hover start."""
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Trigger a repaint on hover end."""
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Trigger a repaint on press."""
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Trigger a repaint on release."""
        self.update()
        super().mouseReleaseEvent(event)
