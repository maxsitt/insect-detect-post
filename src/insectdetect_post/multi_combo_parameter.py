"""Multi-select combo box parameter widget for qt-parameters forms.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

qt-parameters ships no multi-select widget, so this fills the gap for list-valued config
fields. It behaves like its ComboParameter, but every entry carries a checkbox and value()
returns a tuple of the checked values instead of a single value.

Subclassing ParameterWidget is what makes ParameterForm pick the widget up:
its values() does an isinstance check and silently skips anything else.

Classes:
    MultiComboParameter: Combo box with checkable entries, whose value is a tuple of values.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from PySide6.QtCore import QEvent, QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import (
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPalette,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QComboBox,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStylePainter,
)
from qt_parameters import ParameterWidget


class _MultiComboBox(QComboBox):
    """Combo box that paints a caller-supplied summary instead of the current entry."""
    _text: str = ""
    _placeholder: str = ""

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the combo box with the summary as its label.

        Mirrors QComboBox.paintEvent(), overriding only the text the label is drawn with.
        """
        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self._text or self._placeholder
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    def display_text(self) -> str:
        return self._text

    def set_display_text(self, text: str) -> None:
        self._text = text
        self.update()

    def set_placeholder(self, placeholder: str) -> None:
        self._placeholder = placeholder
        self.update()


class MultiComboParameter(ParameterWidget):
    """Combo box with checkable entries, whose value is a tuple of values.

    The popup stays open while entries are toggled, by mouse or with Space, and the combo box
    shows the current selection as a comma-separated summary.

    An optional exclusive value can be set, for entries like 'all' or 'none' that are mutually
    exclusive with every other entry. Checking it clears the rest, and checking anything else
    clears it. A minimum selection count can be required, below which entries stay checked.
    """
    value_changed = Signal(tuple)

    _value: tuple[Any, ...] = ()
    _default: tuple[Any, ...] = ()
    _items: tuple[tuple[Any, Any], ...] = ()
    _exclusive: Any = None
    _placeholder: str = "None"
    _min_selection: int = 0
    _pressed_in_popup: bool = False
    # Guards against reacting to check state changes this widget made itself
    # The model's signals cannot simply be blocked instead: that would also
    # suppress 'dataChanged', which the popup view needs to repaint the checkboxes
    _updating: bool = False

    def _init_ui(self) -> None:
        self._model = QStandardItemModel(self)
        self._model.itemChanged.connect(self._item_changed)

        self.combo = _MultiComboBox()
        self.combo.setModel(self._model)
        self.combo.set_placeholder(self._placeholder)
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        view = self.combo.view()
        # Set the item view delegate explicitly to avoid drawing a non-editable combo box's
        # popup as a menu, which would render the check indicators at larger menu size
        view.setItemDelegate(QStyledItemDelegate(self.combo))
        view.viewport().installEventFilter(self)  # mouse clicks on entries
        view.installEventFilter(self)  # keyboard toggling

        self._layout.addWidget(self.combo)
        self.setFocusProxy(self.combo)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Toggle the current entry and keep the popup open, instead of closing on selection."""
        view = self.combo.view()

        if watched is view and event.type() == QEvent.Type.Show:
            self._pressed_in_popup = False

        if watched is view.viewport() and isinstance(event, QMouseEvent):
            if event.type() == QEvent.Type.MouseButtonPress:
                self._pressed_in_popup = True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                # A release without a press inside the popup belongs to the click that opened
                # it, which Qt reports over the current entry and would otherwise toggle
                if self._pressed_in_popup:
                    # pos() rather than position(): the latter is Qt6-only, and this widget
                    # is meant to stay usable under the Qt5 bindings qt-parameters supports
                    self._toggle_index(view.indexAt(event.pos()))
                self._pressed_in_popup = False
                return True

        if (
            watched is view
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Select)
        ):
            self._toggle_index(view.currentIndex())
            return True

        return super().eventFilter(watched, event)

    def exclusive_value(self) -> Any:
        return self._exclusive

    def set_exclusive_value(self, value: Any) -> None:
        """Set the entry that cannot be combined with any other (e.g. 'all')."""
        self._exclusive = value

    def items(self) -> tuple[tuple[Any, Any], ...]:
        return self._items

    def set_items(self, items: Collection) -> None:
        """Set the selectable entries, as a mapping of label to value or a plain sequence."""
        if isinstance(items, Mapping):
            pairs = tuple(items.items())
        else:
            pairs = tuple(i if isinstance(i, tuple) else (i, i) for i in items)

        self._items = pairs
        self._refresh_items()
        # Re-applied so a value that is no longer selectable is dropped rather than kept
        self.set_value(self._value)

    def min_selection(self) -> int:
        return self._min_selection

    def set_min_selection(self, count: int) -> None:
        """Set how many entries must stay checked, enforced while the user toggles entries.

        set_value() is not restricted by this, so a form can still be populated with fewer
        entries; only unchecking past the minimum is refused.
        """
        self._min_selection = count

    def placeholder(self) -> str:
        return self._placeholder

    def set_placeholder(self, placeholder: str) -> None:
        """Set the text shown while nothing is selected."""
        self._placeholder = placeholder
        self.combo.set_placeholder(placeholder)

    def value(self) -> tuple[Any, ...]:
        return super().value()

    def set_value(self, value: Sequence[Any] | None) -> None:
        """Set the checked entries, ignoring duplicates and values that are not selectable."""
        super().set_value(self._normalized(value))
        self._sync_check_states(self._value)
        self._refresh_text()

    def _checked_values(self) -> tuple[Any, ...]:
        """Return the values of all currently checked entries, in entry order."""
        return tuple(
            data for row, (_label, data) in enumerate(self._items)
            if (item := self._model.item(row)) is not None
            and item.checkState() == Qt.CheckState.Checked
        )

    def _item_changed(self, item: QStandardItem) -> None:
        """Recompute the value after an entry was checked or unchecked."""
        if self._updating:
            return
        if not 0 <= item.row() < len(self._items):
            return

        data = self._items[item.row()][1]
        if (
            self._exclusive is not None
            and item.checkState() == Qt.CheckState.Checked
            and data != self._exclusive
        ):
            # Checking any other entry clears the exclusive one; set_value() handles the
            # reverse, collapsing the selection as soon as the exclusive entry is checked
            self.set_value(tuple(v for v in self._checked_values() if v != self._exclusive))
            return

        values = self._normalized(self._checked_values())
        if len(values) < self._min_selection:
            self._sync_check_states(self._value)
            return
        self.set_value(values)

    def _normalized(self, value: Sequence[Any] | None) -> tuple[Any, ...]:
        """Drop unknown and duplicate entries, keeping the given order.

        Unknown values are discarded for the same reason ComboParameter resolves them to
        None: a value with no entry to represent it could neither be seen nor deselected.
        The given order is preserved rather than reordered to match the entries, so that a
        value loaded from a form is returned unchanged and does not read as an edit.
        """
        known = {data for _label, data in self._items}
        normalized = dict.fromkeys(v for v in (value or ()) if v in known)
        if self._exclusive is not None and self._exclusive in normalized:
            return (self._exclusive,)
        return tuple(normalized)

    def _refresh_items(self) -> None:
        """Rebuild the model from the current items, without reacting to the changes."""
        self._updating = True
        try:
            self._model.clear()
            for label, _data in self._items:
                item = QStandardItem(str(label))
                item.setCheckable(True)
                item.setEditable(False)
                item.setCheckState(Qt.CheckState.Unchecked)
                self._model.appendRow(item)
        finally:
            self._updating = False

    def _refresh_text(self) -> None:
        """Show the current selection as a comma-separated summary."""
        self.combo.set_display_text(
            ", ".join(label for label, data in self._items if data in self._value)
        )

    def _sync_check_states(self, value: Sequence[Any]) -> None:
        """Set the check state of every entry to match value."""
        self._updating = True
        try:
            for row, (_label, data) in enumerate(self._items):
                if item := self._model.item(row):
                    item.setCheckState(
                        Qt.CheckState.Checked if data in value else Qt.CheckState.Unchecked
                    )
        finally:
            self._updating = False

    def _toggle_index(self, index: QModelIndex) -> None:
        """Flip the check state of the entry at index, if it is one that can be toggled."""
        if not index.isValid():
            return
        item = self._model.itemFromIndex(index)
        if item is None or not item.isEnabled():
            return
        item.setCheckState(
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
