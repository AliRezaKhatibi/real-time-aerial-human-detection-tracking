from __future__ import annotations
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen, QBrush, QPainter, QPixmap, QCursor
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView


class EditableBoxItem(QGraphicsRectItem):
    changed = Signal() if False else None  # documentation marker; QGraphicsItem is not QObject

    HANDLE = 8.0

    def __init__(self, rect: QRectF, confidence=None, source="manual"):
        super().__init__(rect)
        self.confidence = confidence
        self.source = source
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        base = QColor("#38bdf8") if source == "model" else QColor("#22c55e")
        self.setPen(QPen(base, 2.0))
        fill = QColor(base); fill.setAlpha(28)
        self.setBrush(QBrush(fill))
        self._resize_handle = None
        self._press_pos = None
        self._orig_rect = None

    def _handles(self):
        r = self.rect()
        s = self.HANDLE
        pts = {
            "tl": r.topLeft(), "tr": r.topRight(), "bl": r.bottomLeft(), "br": r.bottomRight(),
            "tm": QPointF(r.center().x(), r.top()), "bm": QPointF(r.center().x(), r.bottom()),
            "ml": QPointF(r.left(), r.center().y()), "mr": QPointF(r.right(), r.center().y()),
        }
        return {k: QRectF(p.x()-s/2, p.y()-s/2, s, s) for k,p in pts.items()}

    def _handle_at(self, pos: QPointF):
        for name, rr in self._handles().items():
            if rr.contains(pos):
                return name
        return None

    def hoverMoveEvent(self, event):
        h = self._handle_at(event.pos()) if self.isSelected() else None
        cursors = {
            "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
            "tm": Qt.CursorShape.SizeVerCursor, "bm": Qt.CursorShape.SizeVerCursor,
            "ml": Qt.CursorShape.SizeHorCursor, "mr": Qt.CursorShape.SizeHorCursor,
        }
        self.setCursor(cursors.get(h, Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        self._resize_handle = self._handle_at(event.pos()) if self.isSelected() else None
        self._press_pos = event.pos()
        self._orig_rect = QRectF(self.rect())
        if self._resize_handle:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._resize_handle:
            super().mouseMoveEvent(event)
            return
        p = event.pos()
        r = QRectF(self._orig_rect)
        h = self._resize_handle
        if "l" in h: r.setLeft(p.x())
        if "r" in h: r.setRight(p.x())
        if "t" in h: r.setTop(p.y())
        if "b" in h: r.setBottom(p.y())
        if h == "tm": r.setTop(p.y())
        if h == "bm": r.setBottom(p.y())
        if h == "ml": r.setLeft(p.x())
        if h == "mr": r.setRight(p.x())
        r = r.normalized()
        if r.width() >= 2 and r.height() >= 2:
            self.setRect(r)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._resize_handle = None
        self._press_pos = None
        self._orig_rect = None
        super().mouseReleaseEvent(event)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        label = "person"
        if self.confidence is not None:
            label += f" {self.confidence:.2f}"
        painter.save()
        painter.setPen(QColor("#ffffff"))
        painter.setBrush(QColor(15, 23, 42, 210))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 8
        th = fm.height() + 4
        r = self.rect()
        tag = QRectF(r.left(), max(0.0, r.top() - th), tw, th)
        painter.drawRect(tag)
        painter.drawText(tag.adjusted(4, 1, -2, -1), Qt.AlignmentFlag.AlignVCenter, label)
        painter.restore()
        if self.isSelected():
            painter.setPen(QPen(QColor("#facc15"), 1.0))
            painter.setBrush(QBrush(QColor("#facc15")))
            for rr in self._handles().values():
                painter.drawRect(rr)

    def box_dict(self):
        sr = self.sceneBoundingRect().normalized()
        return {
            "x1": float(sr.left()), "y1": float(sr.top()),
            "x2": float(sr.right()), "y2": float(sr.bottom()),
            "label": "person", "source": self.source,
            **({"confidence": float(self.confidence)} if self.confidence is not None else {}),
        }


class AnnotationView(QGraphicsView):
    boxesChanged = Signal()
    selectionChangedSignal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#070b13"))
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.scene_obj.selectionChanged.connect(self.selectionChangedSignal.emit)
        self.pixmap_item = None
        self.image_size = None
        self.draw_mode = False
        self._drawing = False
        self._start_scene = None
        self._temp = None

    def load_image(self, path: Path, boxes: list[dict]):
        self.scene_obj.clear()
        pix = QPixmap(str(path))
        if pix.isNull():
            raise RuntimeError(f"Could not open image: {path}")
        self.pixmap_item = self.scene_obj.addPixmap(pix)
        self.pixmap_item.setZValue(-100)
        self.image_size = (pix.width(), pix.height())
        self.scene_obj.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        for b in boxes:
            rect = QRectF(float(b["x1"]), float(b["y1"]), float(b["x2"])-float(b["x1"]), float(b["y2"])-float(b["y1"]))
            item = EditableBoxItem(rect, b.get("confidence"), b.get("source", "manual"))
            self.scene_obj.addItem(item)
        self.fitInView(self.scene_obj.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def box_items(self):
        return [i for i in self.scene_obj.items() if isinstance(i, EditableBoxItem)]

    def current_boxes(self):
        if not self.image_size:
            return []
        w, h = self.image_size
        out = []
        for item in self.box_items():
            b = item.box_dict()
            b["x1"] = max(0.0, min(w, b["x1"]))
            b["y1"] = max(0.0, min(h, b["y1"]))
            b["x2"] = max(0.0, min(w, b["x2"]))
            b["y2"] = max(0.0, min(h, b["y2"]))
            if b["x2"] > b["x1"] and b["y2"] > b["y1"]:
                out.append(b)
        return out

    def set_draw_mode(self, enabled: bool):
        self.draw_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def delete_selected_boxes(self):
        removed = 0
        for item in list(self.scene_obj.selectedItems()):
            if isinstance(item, EditableBoxItem):
                self.scene_obj.removeItem(item)
                removed += 1
        if removed:
            self.boxesChanged.emit()
        return removed

    def add_box(self, rect: QRectF, confidence=None, source="manual"):
        if rect.width() < 3 or rect.height() < 3:
            return
        item = EditableBoxItem(rect.normalized(), confidence, source)
        self.scene_obj.addItem(item)
        item.setSelected(True)
        self.boxesChanged.emit()

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1/1.18
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self.draw_mode and event.button() == Qt.MouseButton.LeftButton:
            p = self.mapToScene(event.position().toPoint())
            if self.scene_obj.sceneRect().contains(p):
                self._drawing = True
                self._start_scene = p
                self._temp = QGraphicsRectItem(QRectF(p, p))
                self._temp.setPen(QPen(QColor("#38bdf8"), 2.0, Qt.PenStyle.DashLine))
                self.scene_obj.addItem(self._temp)
                event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._temp:
            p = self.mapToScene(event.position().toPoint())
            self._temp.setRect(QRectF(self._start_scene, p).normalized())
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            p = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._start_scene, p).normalized().intersected(self.scene_obj.sceneRect())
            if self._temp:
                self.scene_obj.removeItem(self._temp)
            self._temp = None; self._drawing = False
            self.add_box(rect)
            event.accept(); return
        super().mouseReleaseEvent(event)
