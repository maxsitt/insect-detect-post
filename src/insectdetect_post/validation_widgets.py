"""Qt widgets and models of the validation window.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Classes:
    ThumbnailCache: Background thumbnail decoding with an LRU pixmap cache.
    CropGridModel: List model over the crops of one track (lazy thumbnails).
    CropGridView: Icon-mode list view with zoom.
    SingleImageView: One crop at integer magnification with a caption.
    RowStatus: State of one track row (verdict, flags, links, tags, notes).
    TrackListModel: List model over the validation queue with filters, search and sorting.
    TaxonCombo: Editable combo box with a type-ahead completer for one rank.

Functions:
    parse_search(): Terms of a track list search.
    matches_search(): Whether a track matches every search term.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from PySide6.QtCore import (
    QAbstractListModel,
    QEvent,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QPoint,
    QPointF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QImageReader,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QResizeEvent,
    QShowEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QCompleter,
    QFrame,
    QLabel,
    QListView,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from insectdetect_post.constants import COLOR_FLAG_LOW_QUALITY, COLOR_FLAG_SAME_INDIVIDUAL, COLOR_VERDICT
from insectdetect_post.validation_session import DIFFERENT_TAXA_IN_CROPS, DIFFERENT_TAXA_IN_TRACK
from insectdetect_post.validation_taxonomy import INDET_LABEL, VALIDATION_RANKS

# Display names of the verdict types and of the multiple-individuals types (track list, panel, statistics)
VERDICT_LABELS = {
    "agree": "Agree", "partial": "Partially confirmed", "corrected": "Corrected", "false_positive": "False positive",
    "indeterminate": "Indeterminate", "not_assessable": "Not assessable",
}
MULTIPLE_INDIVIDUALS_LABELS = {
    "same_taxon_in_crops": "Same taxon in crops", DIFFERENT_TAXA_IN_CROPS: "Different taxa in crops",
    DIFFERENT_TAXA_IN_TRACK: "Different taxa in track",
}

# Thumbnails: longest side they are decoded to
_THUMBNAIL_MAX_SIDE = 1024

# Thumbnails: memory budget of the cache (bytes)
_THUMBNAIL_CACHE_BYTES = 150 * 1024 * 1024

# Crop grid: cell sizes (px) of the zoom steps below the crop size (above: integer magnifications)
GRID_ZOOM_STEPS = [64, 96, 128, 160, 192, 256, 320, 384]

# Crop grid: zoom ('px', n) fits crops into an n px cell, ('x', k) shows them at k x their pixels
GridZoom = tuple[str, int]

# Crop grid: memory budget of the scaled icons (bytes)
_ICON_CACHE_BYTES = 80 * 1024 * 1024

# Single view: magnifications (pixel replication)
SINGLE_ZOOM_STEPS = (1, 2, 3, 4, 6, 8, 10)

# Single view: zoom to the largest magnification at which the crop fits the view
FIT = "fit"

# Track list search: field words, as printed in the rows ('ID 140', 'S3', 'D1')
_SEARCH_FIELDS = {"d": "device", "dev": "device", "device": "device", "s": "session", "session": "session",
                  "id": "track", "track": "track"}
_SEARCH_WORD_NUMBER = re.compile(r"^([a-z]+)(\d+)$")

# Track list: parts of a row that show its verdict (text, tooltip, color); only these look up the status
_STATUS_ROLES = frozenset({Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole, Qt.ItemDataRole.ForegroundRole})

# Track list: glyph per verdict type (None = not validated)
_VERDICT_GLYPHS: dict[str | None, str] = {
    None: "○", "agree": "✓", "partial": "◐", "corrected": "✎", "false_positive": "✕",
    # The flag mark next to the glyph tells them apart ('? ▾', '? ⁂')
    "indeterminate": "?", "not_assessable": "?",
}

# Track list: marks of an individual's first track and of the tracks continuing it
_LINK_GLYPH_FIRST = "↱"
LINK_GLYPH_NEXT = "↳"

# Track list: opacity (0-255) of the tint on the shown track's link candidates
_LINK_CANDIDATE_ALPHA = 80


class _ThumbnailSignals(QObject):
    """Signals of the decode tasks (a QRunnable cannot emit)."""

    loaded = Signal(str, QImage, int)  # path, image, longer side of the file (px)


class _ThumbnailTask(QRunnable):
    """Decode one image, scaled to a maximum side, in a worker thread."""

    def __init__(self, path: str, max_side: int, signals: _ThumbnailSignals) -> None:
        super().__init__()
        self._path = path
        self._max_side = max_side
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        reader = QImageReader(self._path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > self._max_side:
            reader.setScaledSize(size.scaled(self._max_side, self._max_side, Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        self._signals.loaded.emit(self._path, image, max(size.width(), size.height()) if size.isValid() else 0)


class ThumbnailCache(QObject):
    """Background thumbnail decoding with an LRU pixmap cache.

    Signals:
        thumbnail_ready(str): A thumbnail was decoded (or failed).
    """
    thumbnail_ready = Signal(str)

    def __init__(self, max_side: int = _THUMBNAIL_MAX_SIDE, capacity_bytes: int = _THUMBNAIL_CACHE_BYTES,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._max_side = max_side
        self._capacity_bytes = capacity_bytes
        self._bytes = 0
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._native_side: dict[str, int] = {}  # longer side (px) of the decoded crops
        self._failed: set[str] = set()
        self._pending: set[str] = set()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(2, min(6, QThreadPool.globalInstance().maxThreadCount())))
        self._signals = _ThumbnailSignals()
        self._signals.loaded.connect(self._on_loaded)

    def get(self, path: str) -> QPixmap | None:
        """Return the cached pixmap, or None (decoding is then scheduled)."""
        pixmap = self._cache.get(path)
        if pixmap is not None:
            self._cache.move_to_end(path)
            return pixmap
        if path not in self._failed:
            self.request(path)
        return None

    def failed(self, path: str) -> bool:
        """Return whether decoding failed (missing or corrupt file)."""
        return path in self._failed

    def native_side(self, path: str) -> int:
        """Return the longer side of the crop file in pixels, 0 if not decoded yet."""
        return self._native_side.get(path, 0)

    def request(self, path: str) -> None:
        """Schedule decoding of a path unless cached, failed or pending."""
        if path in self._cache or path in self._failed or path in self._pending:
            return
        self._pending.add(path)
        self._pool.start(_ThumbnailTask(path, self._max_side, self._signals))

    def prefetch(self, paths: list[str]) -> None:
        """Schedule decoding of several paths."""
        for path in paths:
            self.request(path)

    def cancel_pending(self) -> None:
        """Drop decode tasks that have not started."""
        self._pool.clear()
        self._pending.clear()

    def _on_loaded(self, path: str, image: QImage, native_side: int) -> None:
        """Cache a decoded thumbnail (evicting the least recently used) or remember the failure."""
        self._pending.discard(path)
        if image.isNull():
            self._failed.add(path)
        else:
            self._native_side[path] = native_side or max(image.width(), image.height())
            pixmap = QPixmap.fromImage(image)
            old = self._cache.pop(path, None)
            if old is not None:
                self._bytes -= _pixmap_bytes(old)
            self._cache[path] = pixmap
            self._bytes += _pixmap_bytes(pixmap)
            while self._bytes > self._capacity_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._bytes -= _pixmap_bytes(evicted)
        self.thumbnail_ready.emit(path)


class CropGridModel(QAbstractListModel):
    """List model over the crops of one track, with lazily decoded thumbnails.

    Pixmaps are built in physical pixels. ('px', n) only reduces crops larger than the cell,
    ('x', k) replicates pixels in a cell sized to the largest crop.

    Signals:
        max_side_changed(int): The largest crop side (px) of the track grew.
    """

    PathRole = Qt.ItemDataRole.UserRole + 1
    PixmapRole = Qt.ItemDataRole.UserRole + 2  # painted by _CropDelegate
    max_side_changed = Signal(int)

    def __init__(self, cache: ThumbnailCache, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cache = cache
        self._rows: list[dict[str, Any]] = []
        self._path_to_rows: dict[str, list[int]] = {}
        self._zoom: GridZoom = ("px", GRID_ZOOM_STEPS[2])
        self._dpr = 1.0
        self._max_side = 0
        self._cell = GRID_ZOOM_STEPS[2]
        self._icons: OrderedDict[str, tuple[QPixmap, int]] = OrderedDict()
        self._icon_bytes = 0
        self._placeholder_icon = QIcon()
        self._loading = _placeholder_pixmap(96, "…", QColor("#424242"))
        self._missing = _placeholder_pixmap(96, "missing", QColor("#5D4037"))
        self._update_placeholder()
        cache.thumbnail_ready.connect(self._on_thumbnail_ready)

    def set_images(self, rows: list[dict[str, Any]]) -> None:
        """Replace the images, each row needs 'path', 'label' and 'tooltip'."""
        self.beginResetModel()
        self._rows = rows
        self._path_to_rows = {}
        self._clear_icons()
        for i, row in enumerate(rows):
            self._path_to_rows.setdefault(row["path"], []).append(i)
        self._max_side = max((self._cache.native_side(row["path"]) for row in rows), default=0)
        self._update_placeholder()
        self.endResetModel()
        self._cache.prefetch([row["path"] for row in rows[:200]])

    def refresh_rows(self) -> None:
        """Repaint labels and tooltips after the rows were edited in place."""
        if self._rows:
            self.dataChanged.emit(self.index(0), self.index(len(self._rows) - 1),
                                  [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole])

    @property
    def max_side(self) -> int:
        """Longer side (px) of the largest decoded crop of the track, 0 if none."""
        return self._max_side

    def set_zoom(self, zoom: GridZoom, dpr: float = 1.0) -> None:
        """Set the grid zoom and the screen's device pixel ratio."""
        if zoom != self._zoom or dpr != self._dpr:
            self._zoom = zoom
            self._dpr = dpr
            self._clear_icons()
        self._update_placeholder()

    def cell_side(self) -> int:
        """Return the side (logical px) of a grid cell."""
        mode, value = self._zoom
        if mode == "px":
            return value
        side = self._max_side or GRID_ZOOM_STEPS[0]  # until the first crop is decoded
        return max(int(-(-value * side // self._dpr)), GRID_ZOOM_STEPS[0])  # ceil, at least the smallest step

    def _update_placeholder(self) -> None:
        """Rebuild the transparent decoration that reserves the cell area when the cell size changed."""
        cell = self.cell_side()
        if cell != self._cell or self._placeholder_icon.isNull():
            self._cell = cell
            blank = QPixmap(cell, cell)
            blank.fill(Qt.GlobalColor.transparent)
            self._placeholder_icon = QIcon(blank)  # reserves the decoration area

    def _clear_icons(self) -> None:
        """Drop the zoomed pixmaps (after a zoom change)."""
        self._icons = OrderedDict()
        self._icon_bytes = 0

    def row(self, index: int) -> dict[str, Any] | None:
        """Return the image row at 'index', or None."""
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return row.get("label", "")
        if role == Qt.ItemDataRole.ToolTipRole:
            return row.get("tooltip", "")
        if role == Qt.ItemDataRole.ForegroundRole:
            return QColor(row["foreground"]) if row.get("foreground") else None
        if role == self.PathRole:
            return row["path"]
        if role == Qt.ItemDataRole.DecorationRole:
            return self._placeholder_icon
        if role == self.PixmapRole:
            pixmap = self._cache.get(row["path"])
            if pixmap is None:
                return self._missing if self._cache.failed(row["path"]) else self._loading
            return self._fit(row["path"], pixmap)
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(self._cell + 8, self._cell + 22)
        return None

    def _fit(self, path: str, pixmap: QPixmap) -> QPixmap:
        """Return the thumbnail pixmap for the current zoom (cached), tagged with the device pixel ratio."""
        entry = self._icons.get(path)
        if entry is not None:
            self._icons.move_to_end(path)
            return entry[0]
        mode, value = self._zoom
        if mode == "px":
            limit = round(value * self._dpr)
            if max(pixmap.width(), pixmap.height()) > limit:
                pixmap = pixmap.scaled(limit, limit, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
            elif self._dpr != 1.0:
                pixmap = pixmap.copy()  # keep the cache entry untouched
        elif value > 1:
            pixmap = pixmap.scaled(pixmap.width() * value, pixmap.height() * value,
                                   Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        elif self._dpr != 1.0:
            pixmap = pixmap.copy()
        pixmap.setDevicePixelRatio(self._dpr)
        size = _pixmap_bytes(pixmap)
        self._icons[path] = (pixmap, size)
        self._icon_bytes += size
        while self._icon_bytes > _ICON_CACHE_BYTES and len(self._icons) > 1:
            _, (_, evicted) = self._icons.popitem(last=False)
            self._icon_bytes -= evicted
        return pixmap

    def _on_thumbnail_ready(self, path: str) -> None:
        """Repaint the cells of a decoded crop and track the largest crop side."""
        rows = self._path_to_rows.get(path, [])
        if not rows:
            return
        entry = self._icons.pop(path, None)
        if entry is not None:
            self._icon_bytes -= entry[1]
        side = self._cache.native_side(path)
        if side > self._max_side:
            self._max_side = side
            self.max_side_changed.emit(side)
        for i in rows:
            idx = self.index(i)
            self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])


class _CropDelegate(QStyledItemDelegate):
    """Paints the crop centered in the decoration area on whole physical pixels.

    Item views would resample decorations on displays with fractional scaling.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex | QPersistentModelIndex) -> None:
        super().paint(painter, option, index)  # background, selection, label
        pixmap = index.data(CropGridModel.PixmapRole)
        if not isinstance(pixmap, QPixmap) or pixmap.isNull():
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemDecoration, opt, widget)
        size = pixmap.deviceIndependentSize()
        _draw_on_device_pixels(painter, pixmap, rect.x() + (rect.width() - size.width()) / 2,
                               rect.y() + (rect.height() - size.height()) / 2)


class CropGridView(QListView):
    """Icon-mode list view with zoom (Ctrl+wheel, +/-), double-click or Enter opens a crop.

    Signals:
        zoom_requested(int): +1 / -1 zoom step.
        open_requested(int): Row to open in single-image mode.
    """
    zoom_requested = Signal(int)
    open_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(4)
        self.setWordWrap(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.doubleClicked.connect(lambda idx: self.open_requested.emit(idx.row()))
        self.setItemDelegate(_CropDelegate(self))
        self._zoom: GridZoom = ("px", GRID_ZOOM_STEPS[2])
        self.set_zoom(self._zoom)

    def set_icon_side(self, side: int) -> None:
        """Fixed cell size: crops are reduced to fit, never enlarged."""
        self.set_zoom(("px", side))

    def set_zoom(self, zoom: GridZoom) -> None:
        """Apply a grid zoom (see GridZoom) and size the cells accordingly."""
        self._zoom = zoom
        model = self.model()
        side = zoom[1] if zoom[0] == "px" else GRID_ZOOM_STEPS[2]
        if isinstance(model, CropGridModel):
            model.set_zoom(zoom, self.devicePixelRatioF())
            side = model.cell_side()
        self.setIconSize(QSize(side, side))
        self.setGridSize(QSize(side + 12, side + 26))
        if isinstance(model, CropGridModel):
            model.layoutChanged.emit()

    def cell_side(self) -> int:
        """Return the side (logical px) of a grid cell."""
        return self.iconSize().width()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.set_zoom(self._zoom)  # the pixel ratio is known now

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self.zoom_requested.emit(1 if delta > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_requested.emit(1)
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_requested.emit(-1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.currentIndex().isValid():
            self.open_requested.emit(self.currentIndex().row())
        else:
            super().keyPressEvent(event)
            return
        event.accept()


class _PixelCanvas(QWidget):
    """One pixmap on whole physical pixels, or a text placeholder."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._text = "No image"
        self.setMinimumSize(200, 200)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Show a pixmap, the canvas takes its size."""
        self._pixmap = pixmap
        self._text = ""
        size = pixmap.deviceIndependentSize().toSize()
        self.setMinimumSize(0, 0)
        self.resize(size.width() + 2, size.height() + 2)  # slack for the snapping
        self.update()

    def set_text(self, text: str) -> None:
        """Show a text instead of an image."""
        self._pixmap = QPixmap()
        self._text = text
        self.setMinimumSize(200, 200)
        self.resize(200, 200)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        if self._pixmap.isNull():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)
        else:
            _draw_on_device_pixels(painter, self._pixmap, 1.0, 1.0)
        painter.end()


class SingleImageView(QWidget):
    """One crop at an integer magnification (pixel replication), with a caption.

    1 shows one image pixel per screen pixel, FIT the largest factor that fits. Drag to pan.

    Signals:
        zoom_requested(int): +1 / -1 magnification step.
        zoom_resolved(int): Effective magnification after a render.
        navigate(int): +1 / -1 for the next / previous image.
        closed(): Escape pressed.
        play_toggled(): Space pressed.
    """
    zoom_requested = Signal(int)
    zoom_resolved = Signal(int)
    navigate = Signal(int)
    closed = Signal()
    play_toggled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._scroll.viewport().installEventFilter(self)
        self._image_label = _PixelCanvas()
        self._image_label.setCursor(Qt.CursorShape.OpenHandCursor)
        self._image_label.installEventFilter(self)  # the canvas gets wheel events first
        self._scroll.setWidget(self._image_label)
        layout.addWidget(self._scroll, 1)
        self._caption = QLabel("")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setWordWrap(False)  # see _update_caption()
        layout.addWidget(self._caption, 0)
        self._image: QImage | None = None
        self._caption_text = ""
        self._caption_full = ""
        self._fit_viewport = QSize()
        self._zoom: int | str = FIT
        self._effective = SINGLE_ZOOM_STEPS[0]
        self._drag_origin: QPoint | None = None
        self._drag_scroll = (0, 0)
        self._last_canvas_size = QSize()

    def set_image(self, path: str | Path | None, caption: str) -> None:
        """Load a crop at full resolution and show it with a caption."""
        self._image = None
        if path is not None:
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            image = reader.read()
            if not image.isNull():
                self._image = image
        if self._image is None:
            self._image_label.set_text("Image not found" if path else "No image")
        self._caption_text = caption
        self._render()

    @property
    def effective_zoom(self) -> int:
        """Magnification shown (FIT resolved, a reduced crop counts as 1)."""
        return self._effective

    def set_zoom(self, zoom: int | str) -> None:
        """Set the magnification (SINGLE_ZOOM_STEPS or FIT)."""
        self._zoom = zoom if zoom == FIT or zoom in SINGLE_ZOOM_STEPS else FIT
        self._render()

    def _render(self) -> None:
        """Scale the crop to the magnification (FIT resolved for the view size) and update the caption."""
        if self._image is None:
            self._set_scrollbar_policies(QSize(0, 0))
            self._last_canvas_size = QSize()
            self._caption_full = self._caption_text
            self._update_caption()
            return
        dpr = self.devicePixelRatioF()
        self._fit_viewport = self._scroll.maximumViewportSize()
        avail = self._fit_viewport - QSize(2, 2)  # canvas slack
        avail_px = QSize(int(avail.width() * dpr), int(avail.height() * dpr))
        if avail_px.width() < 10 or avail_px.height() < 10:
            return
        w, h = self._image.width(), self._image.height()
        zoom = self._zoom
        fitting = [z for z in SINGLE_ZOOM_STEPS if z * w <= avail_px.width() and z * h <= avail_px.height()]
        if zoom == FIT:
            zoom = fitting[-1] if fitting else 1
        self._effective = int(zoom)
        if zoom == 1 and not fitting:
            scale = min(avail_px.width() / w, avail_px.height() / h)  # too large: reduce
            mode = Qt.TransformationMode.SmoothTransformation
            how = f"reduced to {scale:.2f}×"
        else:
            scale = float(zoom)
            mode = Qt.TransformationMode.FastTransformation  # exact pixel replication
            how = "1:1" if zoom == 1 else f"{zoom}× pixels"
        if self._zoom == FIT:
            how = f"fit: {how}"
        if scale == 1.0:
            image = self._image
        else:
            image = self._image.scaled(max(1, round(w * scale)), max(1, round(h * scale)),
                                       Qt.AspectRatioMode.KeepAspectRatio, mode)
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(dpr)  # 1:1 on scaled displays too
        self._set_scrollbar_policies(pixmap.deviceIndependentSize().toSize() + QSize(2, 2))
        self._image_label.set_pixmap(pixmap)
        canvas = self._image_label.size()
        if canvas != self._last_canvas_size:
            # Center on a new size, same-sized crops keep the pan position
            self._last_canvas_size = canvas
            for bar in (self._scroll.horizontalScrollBar(), self._scroll.verticalScrollBar()):
                bar.setValue((bar.minimum() + bar.maximum()) // 2)
        self._caption_full = f"{w} × {h} px   ·   {how}   ·   {self._caption_text}"
        self._update_caption()
        self.zoom_resolved.emit(self._effective)

    def _update_caption(self) -> None:
        """Show the caption on one line, elided if too long (full text as tooltip).

        A wrapping caption would change the image height and with it the fitted magnification.
        """
        width = max(0, self._caption.width() - 4)
        text = self._caption_full
        if width > 40:
            text = self._caption.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, width)
        self._caption.setText(text)
        self._caption.setToolTip(self._caption_full if text != self._caption_full else "")

    def _set_scrollbar_policies(self, canvas: QSize) -> None:
        """Set the scrollbars the canvas needs up front, so centring does not jump."""
        full = self._scroll.maximumViewportSize()
        extent = self._scroll.style().pixelMetric(self._scroll.style().PixelMetric.PM_ScrollBarExtent)
        need_v = canvas.height() > full.height()
        need_h = canvas.width() > full.width() - (extent if need_v else 0)
        need_v = canvas.height() > full.height() - (extent if need_h else 0)
        on, off = Qt.ScrollBarPolicy.ScrollBarAlwaysOn, Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        self._scroll.setHorizontalScrollBarPolicy(on if need_h else off)
        self._scroll.setVerticalScrollBarPolicy(on if need_v else off)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_caption()
        if self._zoom in (FIT, 1):
            self._render()  # depends on the view size
        elif self._image is not None:
            self._set_scrollbar_policies(self._image_label.size())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Handle the wheel here, before the scroll area consumes it
        if watched in (self._scroll.viewport(), self._image_label) and isinstance(event, QWheelEvent):
            self.wheelEvent(event)
            return True
        # The viewport can resize on its own: re-resolve FIT
        if (watched is self._scroll.viewport() and event.type() == QEvent.Type.Resize
                and self._image is not None and self._zoom in (FIT, 1)
                and self._scroll.maximumViewportSize() != self._fit_viewport):
            self._render()
        # Drag to pan
        if watched is self._scroll.viewport() and isinstance(event, QMouseEvent):
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_origin = event.position().toPoint()
                self._drag_scroll = (self._scroll.horizontalScrollBar().value(),
                                     self._scroll.verticalScrollBar().value())
                self._image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.setFocus()
                return True
            if event.type() == event.Type.MouseMove and self._drag_origin is not None:
                delta = event.position().toPoint() - self._drag_origin
                self._scroll.horizontalScrollBar().setValue(self._drag_scroll[0] - delta.x())
                self._scroll.verticalScrollBar().setValue(self._drag_scroll[1] - delta.y())
                return True
            if event.type() == event.Type.MouseButtonRelease:
                self._drag_origin = None
                self._image_label.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if delta:
                self.zoom_requested.emit(1 if delta > 0 else -1)
            event.accept()
            return
        if delta:
            self.navigate.emit(1 if delta < 0 else -1)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.navigate.emit(-1)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.navigate.emit(1)
        elif key == Qt.Key.Key_Space:
            self.play_toggled.emit()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_requested.emit(1)
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_requested.emit(-1)
        elif key == Qt.Key.Key_Escape:
            self.closed.emit()
        else:
            super().keyPressEvent(event)
            return
        event.accept()


class RowStatus(NamedTuple):
    """State of one track row (see TrackListModel.set_status_provider()).

    Attributes:
        verdict: Derived verdict type, None if not validated.
        low_quality: Low image quality flag.
        uncertain: Determination carries 'inc.'.
        multiple: MULTIPLE_INDIVIDUALS_TYPES entry, or None.
        linked: Earlier track IDs this track continues (';'-separated), or None.
        continued: Later track IDs continuing this track (';'-separated), or None.
        individual: Track ID naming the individual, None without links.
        n_individual: Tracks of that individual (0 without links).
        link_candidate: An earlier or later track of the queue could be linked (neither a false positive).
        prefilled_from: Stem of the session the track's entry was pre-filled from, or None.
        prefill_edited: The pre-filled entry was changed since.
        tags: Tags of the track.
        notes: Notes of the track.
    """
    verdict: str | None = None
    low_quality: bool = False
    uncertain: bool = False
    multiple: str | None = None
    linked: str | None = None
    continued: str | None = None
    individual: str | None = None
    n_individual: int = 0
    link_candidate: bool = False
    prefilled_from: str | None = None
    prefill_edited: bool = False
    tags: tuple[str, ...] = ()
    notes: str = ""


def parse_search(query: str) -> list[tuple[str | None, str]]:
    """Return the terms of a search query as '(field, value)', field None for any.

    A field word takes the next term ('ID 140') or its own digits ('S3'). A trailing field word is ignored.
    """
    terms: list[tuple[str | None, str]] = []
    field: str | None = None
    for word in query.casefold().split():
        if word in _SEARCH_FIELDS:
            field = _SEARCH_FIELDS[word]
            continue
        match = _SEARCH_WORD_NUMBER.match(word)
        if field is None and match and match.group(1) in _SEARCH_FIELDS:
            terms.append((_SEARCH_FIELDS[match.group(1)], match.group(2)))
        else:
            terms.append((field, word))
        field = None
    return terms


def matches_search(row: dict[str, Any], terms: list[tuple[str | None, str]], individual: str = "",
                   pred: str = "", remarks: str = "") -> bool:
    """Return whether a track matches every search term.

    Numbers match whole ids: without a field the track, session and individual id (not the device).
    Words match the prediction, the device id and 'remarks' (tags and notes). 'pred' is empty where it must
    stay hidden.
    """
    device_no = str(row.get("device_no") or "")
    device_id = str(row["device_id"]).casefold()
    session_id = str(row["session_id"]).casefold()
    track_id = str(row["track_id"]).casefold()
    link = individual.casefold()
    prediction = pred.casefold()
    remarks = remarks.casefold()
    for field, value in terms:
        if field == "device":
            ok = value == device_no or value in device_id
        elif field == "session":
            ok = value == session_id
        elif field == "track":
            ok = value in (track_id, link)
        elif value.isdigit():
            ok = value in (track_id, session_id, link)
        else:
            ok = value in prediction or value in device_id or value in remarks
        if not ok:
            return False
    return True


class TrackListModel(QAbstractListModel):
    """List model over the validation queue with filters, search and sorting.

    Rows are dicts with 'key', 'track_id', 'device_id', 'device_no', 'session_id', 'setting',
    'pred', 'prob', 'start_time', 'det_conf' and 'duration'. The RowStatus per key comes from
    a provider callable.

    Sort modes: 'chrono' (per device by start time), 'setting' (by camera setting, then chrono),
    'queue' (sample order), TAXON_SORTS and NUMERIC_SORTS.
    """

    KeyRole = Qt.ItemDataRole.UserRole + 1
    FILTERS = ("all", "todo", "link_candidates", "prefilled", "done", "agree", "corrected", "indeterminate",
               "uncertain", "false_positive", "linked", "multiple", "low_quality", "tagged")
    # Sort mode -> (row field, descending); ties see _tie_break()
    NUMERIC_SORTS: ClassVar[dict[str, tuple[str, bool]]] = {
        "prob_desc": ("prob", True),
        "prob_asc": ("prob", False),
        "det_conf_desc": ("det_conf", True),
        "det_conf_asc": ("det_conf", False),
        "duration_desc": ("duration", True),
        "duration_asc": ("duration", False),
    }
    # Sort mode -> rank of the predicted taxon, deeper ranks break ties
    TAXON_SORTS: ClassVar[dict[str, str]] = {f"taxon_{rank}": rank for rank in VALIDATION_RANKS}
    SORTS = ("chrono", "setting", *TAXON_SORTS, *NUMERIC_SORTS, "queue")  # order of the sort drop-down
    CHRONO_SORTS = ("chrono", "setting")  # keep a device's tracks in time order
    DEFAULT_SORT = "chrono"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._all_rows: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []
        self._row_by_key: dict[tuple, int] = {}
        self._status = lambda key: RowStatus()
        self._filter = "all"
        self._terms: list[tuple[str | None, str]] = []
        self._sort = self.DEFAULT_SORT
        self._blind = False
        self._link_candidates: dict[tuple, str] = {}

    def set_link_candidates(self, candidates: dict[tuple, str]) -> None:
        """Set the shown track's link candidates ('key -> tooltip line') and repaint the changed rows.

        data() highlights them in CHRONO_SORTS only, so a change of the sort shows or hides them without a call.
        """
        changed = self._link_candidates.keys() ^ candidates.keys()
        self._link_candidates = dict(candidates)
        for key in changed:
            self.refresh_key(key)

    def set_blind(self, blind: bool, apply: bool = True) -> None:
        """Hide the prediction of tracks not validated yet."""
        self._blind = blind
        if apply:
            self._apply_filter()

    def set_sort(self, mode: str, apply: bool = True) -> None:
        """Set the sort mode. 'apply=False' defers the re-sort."""
        self._sort = mode if mode in self.SORTS else self.DEFAULT_SORT
        if apply:
            self._apply_filter()

    def set_status_provider(self, provider: Callable[[tuple], RowStatus]) -> None:
        """Set 'provider(key) -> RowStatus'."""
        self._status = provider

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        """Replace the rows (see the class docstring) and re-apply filter and sort."""
        self._all_rows = rows
        self._apply_filter()

    def set_filter(self, mode: str, search: str = "") -> None:
        """Set the filter (FILTERS) and the search query (see parse_search())."""
        self._filter = mode if mode in self.FILTERS else "all"
        self._terms = parse_search(search)
        self._apply_filter()

    def _matches(self, row: dict[str, Any]) -> bool:
        """Return whether a row passes the filter and the search."""
        status = self._status(row["key"])
        mode = self._filter
        if mode == "todo" and status.verdict is not None:
            return False
        if mode == "link_candidates" and not status.link_candidate:
            return False
        if mode == "done" and status.verdict is None:
            return False
        if mode in ("agree", "corrected", "false_positive") and status.verdict != mode:
            return False
        if mode == "indeterminate" and status.verdict not in ("indeterminate", "partial", "not_assessable"):
            return False
        if mode == "uncertain" and not status.uncertain:
            return False
        if mode == "linked" and not status.individual:
            return False
        if mode == "multiple" and not status.multiple:
            return False
        if mode == "low_quality" and not status.low_quality:
            return False
        if mode == "prefilled" and not status.prefilled_from:
            return False
        if mode == "tagged" and not status.tags:
            return False
        if self._terms:
            # Blind: the hidden prediction is not searchable
            pred = "" if self._blind and status.verdict is None else (row.get("pred") or "")
            return matches_search(row, self._terms, status.individual or "", pred,
                                  " ".join([*status.tags, status.notes]))
        return True

    def _tie_break(self, row: dict[str, Any]) -> tuple:
        """Return the tie order: prediction (not in blind sessions), then track key."""
        taxon = "" if self._blind else (row.get("pred") or "~").casefold()
        return (taxon, str(row["device_id"]), str(row["session_id"]).zfill(6), str(row["track_id"]).zfill(8))

    def _apply_filter(self) -> None:
        """Rebuild the shown rows from filter, search and sort."""
        self.beginResetModel()
        rows = [row for row in self._all_rows if self._matches(row)]
        if self._sort in self.CHRONO_SORTS:
            # A session has one setting, so each device stays in time order
            rows.sort(key=lambda r: ((r.get("setting") or "~").casefold() if self._sort == "setting" else "",
                                     r["device_id"], str(r.get("start_time") or ""), str(r["session_id"]).zfill(6),
                                     str(r["track_id"]).zfill(8)))
        elif self._sort in self.TAXON_SORTS:
            # The chosen rank first, then the deeper ones
            ranks = VALIDATION_RANKS[VALIDATION_RANKS.index(self.TAXON_SORTS[self._sort]):]
            rows.sort(key=lambda r: (*[(r.get(f"pred_{rank}") or "~").casefold() for rank in ranks],
                                     -(r.get("prob") or 0.0), str(r["device_id"]),
                                     str(r["session_id"]).zfill(6), str(r["track_id"]).zfill(8)))
        elif self._sort in self.NUMERIC_SORTS:
            field, descending = self.NUMERIC_SORTS[self._sort]
            rows.sort(key=self._tie_break)  # stable sort keeps this order for ties
            rows.sort(key=lambda r: -(r.get(field) or 0.0) if descending else (r.get(field) or 0.0))
        self._rows = rows
        self._row_by_key = {row["key"]: i for i, row in enumerate(rows)}
        self.endResetModel()

    def refresh_key(self, key: tuple) -> None:
        """Repaint the row of a key."""
        i = self._row_by_key.get(key)
        if i is not None:
            idx = self.index(i)
            self.dataChanged.emit(idx, idx)

    def row_of_key(self, key: tuple) -> int:
        """Return the shown row of a key, -1 if filtered out."""
        return self._row_by_key.get(key, -1)

    def key_at(self, row: int) -> tuple | None:
        """Return the key of a shown row, or None."""
        return self._rows[row]["key"] if 0 <= row < len(self._rows) else None

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        if role == self.KeyRole:
            return row["key"]
        candidate_tip = self._link_candidates.get(row["key"]) if self._sort in self.CHRONO_SORTS else None
        if role == Qt.ItemDataRole.BackgroundRole:
            if candidate_tip is None:
                return None
            tint = QColor(COLOR_FLAG_SAME_INDIVIDUAL)
            tint.setAlpha(_LINK_CANDIDATE_ALPHA)
            return tint
        if role not in _STATUS_ROLES:
            return None  # only these roles need the status
        status = self._status(row["key"])
        if role == Qt.ItemDataRole.DisplayRole:
            glyph = _VERDICT_GLYPHS.get(status.verdict, "○")
            quality = (" ⁂" if status.multiple else "") + (" ▾" if status.low_quality else "")
            prob = f"{row['prob']:.2f}" if isinstance(row.get("prob"), float) else "-"
            where = f"{_device_label(row)}  ·  S{row['session_id']}  ·  ID {row['track_id']}{_link_label(row, status)}"
            if self._blind and status.verdict is None:
                return f"{glyph}{quality}  {where}  ·  (hidden)"
            return f"{glyph}{quality}  {where}  ·  {row.get('pred') or '-'} ({prob})"
        if role == Qt.ItemDataRole.ToolTipRole:
            pred = "(hidden)" if self._blind and status.verdict is None else (row.get("pred") or "-")
            verdict = VERDICT_LABELS.get(status.verdict, status.verdict) if status.verdict else "not validated"
            return (f"Device: {_device_label(row)} = {row['device_id']}\nSession: {row['session_id']}\n"
                    f"Track ID: {row['track_id']}\n"
                    f"Prediction: {pred}\n"
                    f"Verdict: {verdict}"
                    + ("\nUncertain (inc.)" if status.uncertain else "")
                    + (f"\nIndividual: ID {status.individual} ({status.n_individual} tracks)"
                       if status.individual else "")
                    + (f"\nSame individual as ID {status.linked.replace(';', ', ')}" if status.linked else "")
                    + (f"\nContinued by ID {status.continued.replace(';', ', ')}" if status.continued else "")
                    + (f"\nMultiple individuals: {MULTIPLE_INDIVIDUALS_LABELS.get(status.multiple, status.multiple)}"
                       if status.multiple else "")
                    + ("\nLow image quality" if status.low_quality else "")
                    + (f"\nTags: {', '.join(status.tags)}" if status.tags else "")
                    + (f"\nNotes: {status.notes}" if status.notes else "")
                    + (f"\nPre-filled from {status.prefilled_from}{' (edited)' if status.prefill_edited else ''}"
                       if status.prefilled_from else "")
                    + (f"\n\n{candidate_tip}" if candidate_tip else ""))
        if role == Qt.ItemDataRole.ForegroundRole:
            if status.verdict in COLOR_VERDICT:
                return QColor(COLOR_VERDICT[status.verdict])
            if status.low_quality:
                return QColor(COLOR_FLAG_LOW_QUALITY)
            return None
        return None


class TaxonCombo(QComboBox):
    """Editable combo box with a type-ahead completer for one rank.

    Items are the children of the selected parent ('indet.' first). The completer can search a larger list.

    Signals:
        committed(str): Text committed by choosing an item or finishing the edit.
    """
    committed = Signal(str)

    def __init__(self, rank: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rank = rank
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setMaxVisibleItems(20)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._completer = QCompleter(self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setMaxVisibleItems(15)
        self.setCompleter(self._completer)
        self._last_committed: str | None = None
        self._items: list[str] | None = None
        self._completer_model: Any | None = None
        self.activated.connect(self._on_activated)
        self._completer.activated.connect(self._on_completer_activated)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self._on_editing_finished)
            line_edit.setPlaceholderText(INDET_LABEL)

    def set_choices(self, items: list[str], completer_model: Any | None) -> None:
        """Set the items ('indet.' prepended) and the completer model, no-op if unchanged."""
        if items == self._items and completer_model is self._completer_model:
            return
        self._items = list(items)
        self._completer_model = completer_model
        text = self.currentText()
        self.blockSignals(True)
        self.clear()
        self.addItem(INDET_LABEL)
        if items:
            self.addItems(items)
        if completer_model is not None:
            self._completer.setModel(completer_model)
            self._completer.setModelSorting(QCompleter.ModelSorting.CaseInsensitivelySortedModel)
        else:
            self._completer.setModel(self.model())
            self._completer.setModelSorting(QCompleter.ModelSorting.UnsortedModel)
        self.setEditText(text)
        self.blockSignals(False)

    def set_text(self, text: str | None) -> None:
        """Set the shown text without emitting 'committed'."""
        self.blockSignals(True)
        self.setEditText(text or "")
        self.blockSignals(False)
        self._last_committed = text or ""

    def text(self) -> str:
        """Return the shown text without surrounding spaces."""
        return self.currentText().strip()

    def _commit(self, text: str) -> bool:
        """Emit 'committed' for a changed text, return whether it changed."""
        text = text.strip()
        if text == self._last_committed:
            return False
        self._last_committed = text
        self.committed.emit(text)
        return True

    def _on_activated(self, index: int) -> None:
        """Commit an item chosen from the list."""
        self._commit(self.itemText(index))

    def _on_completer_activated(self, text: str) -> None:
        """Commit a completion chosen from the popup."""
        self.setEditText(text)
        self._commit(text)

    def _on_editing_finished(self) -> None:
        """Commit a typed text when the field loses focus (not while the popup is open)."""
        if (popup := self._completer.popup()) is not None and popup.isVisible():
            return
        self._commit(self.currentText())

    def commit_current(self) -> bool:
        """Commit the current text, False if it was already committed."""
        popup = self._completer.popup()
        if popup is not None and popup.isVisible():
            popup.hide()
        return self._commit(self.currentText())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.set_text(self._last_committed)
            self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)


def _pixmap_bytes(pixmap: QPixmap) -> int:
    """Return the approximate memory of a pixmap (32 bpp)."""
    return pixmap.width() * pixmap.height() * 4


def _placeholder_pixmap(size: int, text: str, color: QColor) -> QPixmap:
    """Return a square placeholder tile with a centered text."""
    pixmap = QPixmap(size, size)
    pixmap.fill(color)
    painter = QPainter(pixmap)
    painter.setPen(QColor("#BDBDBD"))
    painter.drawText(pixmap.rect().adjusted(4, 4, -4, -4),
                     Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
    painter.end()
    return pixmap


def _draw_on_device_pixels(painter: QPainter, pixmap: QPixmap, x: float, y: float) -> None:
    """Draw 'pixmap' at logical (x, y), snapped to whole physical pixels (no resampling)."""
    device = painter.deviceTransform()
    inverse, ok = device.inverted()
    if not ok:
        painter.drawPixmap(QPointF(x, y), pixmap)
        return
    target = device.map(QPointF(x, y))
    snapped = inverse.map(QPointF(round(target.x()), round(target.y())))
    painter.drawPixmap(snapped, pixmap)


def _device_label(row: dict[str, Any]) -> str:
    """Return 'D<n>' of a track's device (the device id if unnumbered)."""
    return f"D{row['device_no']}" if row.get("device_no") else str(row["device_id"])


def _link_label(row: dict[str, Any], status: RowStatus) -> str:
    """Return the individual mark: the track count on the first track, its ID on the others."""
    if not status.individual:
        return ""
    if status.individual == str(row["track_id"]):
        return f" {_LINK_GLYPH_FIRST}×{status.n_individual}"
    return f" {LINK_GLYPH_NEXT}{status.individual}"
