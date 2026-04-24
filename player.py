from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any

try:
    from PySide6.QtCore import QEvent, QObject, QPoint, QSignalBlocker, QSize, QThread, QTimer, Qt, Signal, QUrl
    from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygon, QShortcut
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QListView,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStatusBar,
        QStyle,
        QTableWidget,
        QTableWidgetItem,
        QToolButton,
        QToolTip,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 is required for the desktop player. Install it with: python -m pip install PySide6"
    ) from exc


TAXONOMY = [
    {
        "label": "Core Content",
        "kind": "content",
        "color": "#2f9e44",
    },
    {
        "label": "Intro",
        "kind": "non-content",
        "color": "#1c7ed6",
    },
    {
        "label": "Outro",
        "kind": "non-content",
        "color": "#495057",
    },
    {
        "label": "Advertisement",
        "kind": "non-content",
        "color": "#c92a2a",
    },
    {
        "label": "Self-Promotion",
        "kind": "non-content",
        "color": "#e67700",
    },
    {
        "label": "Recap",
        "kind": "non-content",
        "color": "#f08c00",
    },
    {
        "label": "Transition",
        "kind": "non-content",
        "color": "#5f3dc4",
    },
    {
        "label": "Inactivity",
        "kind": "non-content",
        "color": "#868e96",
    },
    {
        "label": "Filler",
        "kind": "non-content",
        "color": "#a61e4d",
    },
]
EPSILON_SECONDS = 0.04
TIMELINE_TRACK_SIDE_PADDING = 9


@dataclass
class Segment:
    identifier: str
    start: float
    end: float
    label_name: str
    kind: str
    color: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def taxonomy_item_for_label(label_name: str) -> dict[str, str] | None:
    for item in TAXONOMY:
        if item["label"] == label_name:
            return item
    return None


def build_segment_from_payload(payload: dict[str, Any], index: int) -> Segment | None:
    try:
        start = float(payload["start"])
        end = float(payload["end"])
    except (KeyError, TypeError, ValueError):
        return None

    if end <= start:
        return None

    label_name = str(payload.get("label") or "Core Content")
    taxonomy_item = taxonomy_item_for_label(label_name) or {
        "label": label_name,
        "kind": str(payload.get("kind") or "non-content"),
        "color": "#868e96",
    }

    return Segment(
        identifier=str(payload.get("id") or f"segment-{index}-{int(start * 1000)}-{int(end * 1000)}"),
        start=start,
        end=end,
        label_name=str(taxonomy_item["label"]),
        kind=str(payload.get("kind") or taxonomy_item["kind"]),
        color=str(payload.get("color") or taxonomy_item["color"]),
    )


def build_even_segments(duration_seconds: float) -> list[Segment]:
    if duration_seconds <= 0:
        raise ValueError("Video duration must be positive before building segments.")

    segments: list[Segment] = []
    demo_label_order = [
        "Intro",
        "Core Content",
        "Advertisement",
        "Core Content",
        "Recap",
        "Core Content",
        "Transition",
        "Core Content",
        "Self-Promotion",
        "Core Content",
        "Inactivity",
        "Core Content",
        "Filler",
        "Outro",
    ]
    total_labels = len(demo_label_order)
    for index, label_name in enumerate(demo_label_order):
        item = taxonomy_item_for_label(label_name)
        if item is None:
            raise ValueError(f"Unknown demo segment label: {label_name}")
        start = duration_seconds * index / total_labels
        end = duration_seconds * (index + 1) / total_labels
        payload = {
            "id": f"segment-{index}",
            "start": start,
            "end": end,
            "label": item["label"],
            "kind": item["kind"],
            "color": item["color"],
        }
        segment = build_segment_from_payload(payload, index)
        if segment is not None:
            segments.append(segment)
    return segments


def run_video_segmentation(video_path: Path) -> list[Segment]:
    simulated_step_delay_seconds = 0.8

    sleep(simulated_step_delay_seconds)
    duration_seconds = probe_video_duration_seconds(video_path)
    sleep(simulated_step_delay_seconds)
    segments = build_even_segments(duration_seconds)
    sleep(simulated_step_delay_seconds)
    return segments


def probe_video_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    duration_text = completed.stdout.strip()
    duration_seconds = float(duration_text)
    if duration_seconds <= 0:
        raise ValueError(f"Unable to determine a positive duration for {path.name}.")
    return duration_seconds


def format_time(total_seconds: float) -> str:
    safe_value = max(0, int(total_seconds))
    hours, remainder = divmod(safe_value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class PositionSlider(QSlider):
    positionClicked = Signal(int)
    positionDragged = Signal(int)

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._interaction_maximum: int | None = None

    def setInteractionMaximum(self, maximum: int | None) -> None:
        self._interaction_maximum = None if maximum is None else max(self.minimum(), int(maximum))
        if self._interaction_maximum is not None and self.value() > self._interaction_maximum:
            self.setValue(self._interaction_maximum)

    def _value_from_mouse_x(self, x: float) -> int:
        track_width = max(1, self.width() - (TIMELINE_TRACK_SIDE_PADDING * 2))
        relative_x = min(max(x - TIMELINE_TRACK_SIDE_PADDING, 0.0), float(track_width))
        value = QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            int(relative_x),
            track_width,
        )
        if self._interaction_maximum is not None:
            value = min(value, self._interaction_maximum)
        return value

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            value = self._value_from_mouse_x(event.position().x())
            self.setValue(value)
            self.positionClicked.emit(value)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            value = self._value_from_mouse_x(event.position().x())
            self.setValue(value)
            self.positionDragged.emit(value)
            event.accept()
            return
        super().mouseMoveEvent(event)


class SpinnerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step = 0
        self._dot_count = 12
        self.setFixedSize(44, 44)

    def advance(self) -> None:
        self._step = (self._step + 1) % self._dot_count
        self.update()

    def reset(self) -> None:
        self._step = 0
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center_x = self.width() / 2
        center_y = self.height() / 2
        orbit_radius = min(self.width(), self.height()) * 0.3
        dot_radius = 3.0

        for index in range(self._dot_count):
            angle = (360 / self._dot_count) * index
            radians = math.radians(angle)
            x = center_x + orbit_radius * math.cos(radians)
            y = center_y + orbit_radius * math.sin(radians)
            distance = (index - self._step) % self._dot_count
            alpha = max(35, 255 - distance * 18)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(248, 250, 252, alpha))
            painter.drawEllipse(QPoint(int(x), int(y)), int(dot_radius), int(dot_radius))


class PopupAwareComboBox(QComboBox):
    popupShown = Signal()
    popupHidden = Signal()

    def showPopup(self) -> None:  # type: ignore[override]
        self.popupShown.emit()
        super().showPopup()

    def hidePopup(self) -> None:  # type: ignore[override]
        try:
            super().hidePopup()
        finally:
            self.popupHidden.emit()


class SegmentationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, video_path: Path) -> None:
        super().__init__()
        self.video_path = video_path

    def run(self) -> None:
        try:
            segments = run_video_segmentation(self.video_path)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(segments)


class PlayerVideoWidget(QVideoWidget):
    toggleFullScreenRequested = Signal()
    exitFullScreenRequested = Signal()
    togglePlaybackRequested = Signal()
    pointerMoved = Signal(QPoint)
    widgetResized = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        self.pointerMoved.emit(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.togglePlaybackRequested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.widgetResized.emit()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.exitFullScreenRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F:
            self.toggleFullScreenRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class SegmentTimelineWidget(QWidget):
    segmentSelected = Signal(str)
    segmentActivated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._segments: list[Segment] = []
        self._duration_seconds = 0.0
        self._position_seconds = 0.0
        self._selected_segment_id: str | None = None
        self._active_segment_id: str | None = None
        self._hovered_segment_id: str | None = None
        self.setMinimumHeight(78)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def _track_rect(self):
        return self.rect().adjusted(TIMELINE_TRACK_SIDE_PADDING, 8, -TIMELINE_TRACK_SIDE_PADDING, -12)

    def _segment_at_time(self, requested_time: float) -> Segment | None:
        for segment in self._segments:
            if segment.start - EPSILON_SECONDS <= requested_time <= segment.end + EPSILON_SECONDS:
                return segment
        return None

    def _segment_at_position(self, point: QPoint) -> Segment | None:
        track_rect = self._track_rect()
        if not track_rect.contains(point) or self._duration_seconds <= 0:
            return None
        x_ratio = min(max((point.x() - track_rect.left()) / max(1.0, track_rect.width()), 0.0), 1.0)
        return self._segment_at_time(x_ratio * self._duration_seconds)

    def set_state(
        self,
        *,
        segments: list[Segment],
        duration_seconds: float,
        position_seconds: float,
        selected_segment_id: str | None,
        active_segment_id: str | None,
    ) -> None:
        self._segments = segments
        self._duration_seconds = max(duration_seconds, 0.0)
        self._position_seconds = max(position_seconds, 0.0)
        self._selected_segment_id = selected_segment_id
        self._active_segment_id = active_segment_id
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        full_rect = self._track_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e4ddd0"))
        painter.drawRoundedRect(full_rect, 14, 14)

        if not self._segments or self._duration_seconds <= 0:
            painter.setPen(QColor("#6c757d"))
            painter.drawText(full_rect, Qt.AlignmentFlag.AlignCenter, "Load segments to show the timeline")
            return

        for segment in self._segments:
            start_ratio = min(max(segment.start / self._duration_seconds, 0.0), 1.0)
            end_ratio = min(max(segment.end / self._duration_seconds, 0.0), 1.0)
            left = full_rect.left() + int(full_rect.width() * start_ratio)
            right = full_rect.left() + int(full_rect.width() * end_ratio)
            width = max(3, right - left)
            segment_rect = full_rect.adjusted(left - full_rect.left(), 0, -(full_rect.width() - (left - full_rect.left() + width)), 0)

            painter.setBrush(QColor(segment.color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(segment_rect, 10, 10)

            if segment.identifier == self._selected_segment_id:
                painter.setPen(QPen(QColor("#0f766e"), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(segment_rect.adjusted(1, 1, -1, -1), 10, 10)
            elif segment.identifier == self._active_segment_id:
                painter.setPen(QPen(QColor("#264653"), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(segment_rect.adjusted(1, 1, -1, -1), 10, 10)

        playhead_ratio = min(max(self._position_seconds / self._duration_seconds, 0.0), 1.0)
        playhead_x = full_rect.left() + int(full_rect.width() * playhead_ratio)
        painter.setPen(QPen(QColor("#111827"), 3))
        painter.drawLine(playhead_x, full_rect.top() - 4, playhead_x, full_rect.bottom() + 4)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        segment = self._segment_at_position(event.position().toPoint())
        hovered_identifier = segment.identifier if segment else None
        if hovered_identifier != self._hovered_segment_id:
            self._hovered_segment_id = hovered_identifier
            if segment is not None:
                QToolTip.showText(
                    event.globalPosition().toPoint() + QPoint(12, -8),
                    segment.label_name,
                    self,
                )
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._segments or self._duration_seconds <= 0:
            return

        segment = self._segment_at_position(event.position().toPoint())
        if segment is not None:
            self.segmentSelected.emit(segment.identifier)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if not self._segments or self._duration_seconds <= 0:
            return

        segment = self._segment_at_position(event.position().toPoint())
        if segment is not None:
            self.segmentActivated.emit(segment.identifier)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hovered_segment_id = None
        QToolTip.hideText()
        super().leaveEvent(event)


class PlayerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CSCI 576 Segment Player")
        self.resize(1460, 940)

        self.current_video_path: Path | None = None
        self.segments: list[Segment] = []
        self.selected_segment_id: str | None = None
        self.active_segment_id: str | None = None
        self.slider_is_active = False
        self.active_position_slider: PositionSlider | None = None
        self.user_selected_segment = False
        self.fullscreen_overlay_watchers: list[QWidget] = []
        self.fullscreen_popup_open = False
        self.last_fullscreen_cursor_pos = QCursor.pos()
        self.fullscreen_cursor_hidden = False
        self.last_fullscreen_activity_at = monotonic()
        self.fullscreen_idle_hide_seconds = 0.75
        self.processing_overlay_visible = False
        self.processing_thread: QThread | None = None
        self.processing_worker: SegmentationWorker | None = None

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.85)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.video_widget = PlayerVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)
        self.fullscreen_cursor_poll_timer = QTimer(self)
        self.fullscreen_cursor_poll_timer.setInterval(90)
        self.processing_animation_timer = QTimer(self)
        self.processing_animation_timer.setInterval(120)

        self._build_ui()
        self._connect_signals()
        self._apply_styles()
        QApplication.instance().installEventFilter(self)
        self._load_initial_state()

    def _build_ui(self) -> None:
        self.setStatusBar(QStatusBar(self))

        transport_box = QWidget()
        transport_layout = QVBoxLayout(transport_box)
        transport_layout.setContentsMargins(0, 0, 0, 0)
        transport_layout.setSpacing(12)

        source_row = QHBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.setSpacing(18)
        self.video_source_label = QLabel("Video: none")
        self.video_source_label.setObjectName("sourceLabel")
        source_row.addWidget(self.video_source_label)
        source_row.addStretch()
        transport_layout.addLayout(source_row)

        self.video_widget.setMinimumHeight(460)
        transport_layout.addWidget(self.video_widget, stretch=1)
        self._build_processing_overlay()
        self._build_fullscreen_overlay()

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        self.play_pause_button = QToolButton()
        self.play_pause_button.setObjectName("mediaButton")
        self.play_pause_button.setToolTip("Play/Pause")
        self.play_pause_button.setIconSize(QSize(20, 20))
        self.play_pause_button.setAutoRaise(False)
        self.prev_segment_button = QPushButton("Prev Segment")
        self.next_segment_button = QPushButton("Next Segment")
        self.fullscreen_button = QToolButton()
        self.fullscreen_button.setObjectName("mediaButton")
        self.fullscreen_button.setToolTip("Enter Fullscreen")
        self.fullscreen_button.setIconSize(QSize(18, 18))
        self.fullscreen_button.setAutoRaise(False)
        self.content_only_checkbox = QCheckBox("Play Content Only")
        self.content_only_checkbox.setObjectName("inlineCheck")
        self.speed_combo = QComboBox()
        self.speed_combo.setObjectName("inlineCombo")
        self.speed_label = QLabel("Speed")
        self.speed_label.setObjectName("inlineLabel")
        self.volume_label = QLabel("Volume")
        self.volume_label.setObjectName("inlineLabel")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setObjectName("volumeSlider")
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self.audio_output.volume() * 100))
        self.volume_slider.setFixedWidth(120)
        for label, rate in (("0.5x", 0.5), ("0.75x", 0.75), ("1.0x", 1.0), ("1.25x", 1.25), ("1.5x", 1.5), ("2.0x", 2.0)):
            self.speed_combo.addItem(label, rate)
        self.speed_combo.setCurrentIndex(2)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.prev_segment_button)
        controls_layout.addWidget(self.next_segment_button)
        controls_layout.addWidget(self.fullscreen_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.content_only_checkbox)
        controls_layout.addWidget(self.volume_label)
        controls_layout.addWidget(self.volume_slider)
        controls_layout.addWidget(self.speed_label)
        controls_layout.addWidget(self.speed_combo)
        transport_layout.addLayout(controls_layout)
        self._update_content_only_checkbox_icons(self.content_only_checkbox.isChecked())

        self._update_playback_button_icon(self.player.playbackState())
        self._update_fullscreen_button_icon(self.is_fullscreen_active())

        self.timeline_widget = SegmentTimelineWidget()
        transport_layout.addWidget(self.timeline_widget)

        self.position_slider = PositionSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        transport_layout.addWidget(self.position_slider)

        time_row = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00")
        self.segment_badge_label = QLabel("No active segment")
        self.time_label.setObjectName("metricLabel")
        self.segment_badge_label.setObjectName("metricLabel")
        time_row.addWidget(self.time_label)
        time_row.addStretch()
        time_row.addWidget(self.segment_badge_label)
        transport_layout.addLayout(time_row)

        self.metrics_box = QGroupBox("Overview")
        metrics_layout = QGridLayout(self.metrics_box)
        metrics_layout.setContentsMargins(14, 18, 14, 14)
        metrics_layout.setHorizontalSpacing(14)
        metrics_layout.setVerticalSpacing(8)
        self.total_segments_value = QLabel("0")
        self.content_duration_value = QLabel("00:00")
        self.non_content_duration_value = QLabel("00:00")
        self._add_metric(metrics_layout, 0, "Total Segments", self.total_segments_value)
        self._add_metric(metrics_layout, 1, "Content Time", self.content_duration_value)
        self._add_metric(metrics_layout, 2, "Non-Content Time", self.non_content_duration_value)

        self.segment_table = QTableWidget(0, 2)
        self.segment_table.setHorizontalHeaderLabels(["Segment Type", "Time"])
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.segment_table.setSelectionMode(QTableWidget.SingleSelection)
        self.segment_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.segment_table.setAlternatingRowColors(True)
        self.segment_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.segment_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        table_group = QGroupBox("Segment List")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(14, 18, 14, 14)
        table_layout.addWidget(self.segment_table)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(12)
        sidebar_layout.addWidget(self.metrics_box)
        sidebar_layout.addWidget(table_group, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(transport_box)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        container = QWidget()
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(14, 14, 14, 14)
        outer_layout.setSpacing(14)
        outer_layout.addWidget(splitter)
        self.setCentralWidget(container)

        file_menu = self.menuBar().addMenu("&File")
        self.open_video_action = QAction("Open Video", self)
        self.open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(self.open_video_action)

        self.window_fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F), self, self.toggle_fullscreen)
        self.window_exit_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.exit_fullscreen)
        self.fullscreen_toggle_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F), self.video_widget, self.toggle_fullscreen)
        self.fullscreen_toggle_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.fullscreen_exit_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self.video_widget, self.exit_fullscreen)
        self.fullscreen_exit_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.overlay_exit_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self.fullscreen_controls, self.exit_fullscreen)
        self.overlay_exit_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.overlay_fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F), self.fullscreen_controls, self.toggle_fullscreen)
        self.overlay_fullscreen_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)

    def _build_fullscreen_overlay(self) -> None:
        self.fullscreen_controls = QFrame(
            self.video_widget,
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.fullscreen_controls.setObjectName("fullscreenOverlay")
        self.fullscreen_controls.hide()
        self.fullscreen_controls.setMouseTracking(True)
        self.fullscreen_controls.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        overlay_layout = QVBoxLayout(self.fullscreen_controls)
        overlay_layout.setContentsMargins(18, 14, 18, 14)
        overlay_layout.setSpacing(10)

        self.fullscreen_position_slider = PositionSlider(Qt.Orientation.Horizontal, self.fullscreen_controls)
        self.fullscreen_position_slider.setObjectName("fullscreenPositionSlider")
        self.fullscreen_position_slider.setRange(0, 0)
        overlay_layout.addWidget(self.fullscreen_position_slider)

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(12)

        self.fullscreen_play_pause_button = QToolButton()
        self.fullscreen_play_pause_button.setObjectName("overlayMediaButton")
        self.fullscreen_play_pause_button.setIconSize(QSize(20, 20))

        self.fullscreen_time_label = QLabel("00:00 / 00:00")
        self.fullscreen_time_label.setObjectName("overlayTimeLabel")

        self.fullscreen_content_only_checkbox = QToolButton()
        self.fullscreen_content_only_checkbox.setObjectName("overlayToggleButton")
        self.fullscreen_content_only_checkbox.setText("Play Content Only")
        self.fullscreen_content_only_checkbox.setCheckable(True)
        self.fullscreen_content_only_checkbox.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.fullscreen_content_only_checkbox.setIconSize(QSize(18, 18))
        self.fullscreen_content_only_checkbox.setCursor(Qt.CursorShape.ArrowCursor)

        self.fullscreen_volume_label = QLabel("Volume")
        self.fullscreen_volume_label.setObjectName("overlayInlineLabel")
        self.fullscreen_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.fullscreen_volume_slider.setObjectName("overlayVolumeSlider")
        self.fullscreen_volume_slider.setRange(0, 100)
        self.fullscreen_volume_slider.setValue(int(self.audio_output.volume() * 100))
        self.fullscreen_volume_slider.setFixedWidth(140)

        self.fullscreen_speed_label = QLabel("Speed")
        self.fullscreen_speed_label.setObjectName("overlayInlineLabel")
        self.fullscreen_speed_combo = PopupAwareComboBox()
        self.fullscreen_speed_combo.setObjectName("overlayCombo")
        self.fullscreen_speed_combo.setView(QListView())
        self.fullscreen_speed_combo.view().setObjectName("overlayComboPopup")
        self.fullscreen_speed_combo.setMaxVisibleItems(6)
        self.fullscreen_speed_combo.popupShown.connect(self.on_fullscreen_popup_shown)
        self.fullscreen_speed_combo.popupHidden.connect(self.on_fullscreen_popup_hidden)
        for label, rate in (("0.5x", 0.5), ("0.75x", 0.75), ("1.0x", 1.0), ("1.25x", 1.25), ("1.5x", 1.5), ("2.0x", 2.0)):
            self.fullscreen_speed_combo.addItem(label, rate)
        self.fullscreen_speed_combo.setCurrentIndex(2)

        self.fullscreen_exit_button = QToolButton()
        self.fullscreen_exit_button.setObjectName("overlayMediaButton")
        self.fullscreen_exit_button.setIconSize(QSize(18, 18))
        self.fullscreen_exit_button.setToolTip("Exit Fullscreen")

        bottom_row.addWidget(self.fullscreen_play_pause_button)
        bottom_row.addWidget(self.fullscreen_time_label)
        bottom_row.addStretch()
        bottom_row.addWidget(self.fullscreen_content_only_checkbox)
        bottom_row.addWidget(self.fullscreen_volume_label)
        bottom_row.addWidget(self.fullscreen_volume_slider)
        bottom_row.addWidget(self.fullscreen_speed_label)
        bottom_row.addWidget(self.fullscreen_speed_combo)
        bottom_row.addWidget(self.fullscreen_exit_button)
        overlay_layout.addLayout(bottom_row)

        self.fullscreen_overlay_watchers = [
            self.fullscreen_controls,
            self.fullscreen_position_slider,
            self.fullscreen_play_pause_button,
            self.fullscreen_time_label,
            self.fullscreen_content_only_checkbox,
            self.fullscreen_volume_label,
            self.fullscreen_volume_slider,
            self.fullscreen_speed_label,
            self.fullscreen_speed_combo,
            self.fullscreen_speed_combo.view(),
            self.fullscreen_exit_button,
        ]
        for widget in self.fullscreen_overlay_watchers:
            widget.installEventFilter(self)
            widget.setMouseTracking(True)

        self.update_fullscreen_overlay_geometry()

    def _build_processing_overlay(self) -> None:
        self.processing_overlay = QFrame(
            self.video_widget,
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.processing_overlay.setObjectName("processingOverlay")
        self.processing_overlay.hide()

        layout = QVBoxLayout(self.processing_overlay)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)

        self.processing_spinner = SpinnerWidget()
        self.processing_spinner.setObjectName("processingSpinner")
        self.processing_label = QLabel("Processing...")
        self.processing_label.setObjectName("processingOverlayLabel")
        self.processing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.processing_spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.processing_label)

        self.update_processing_overlay_geometry()

    def _add_metric(self, layout: QGridLayout, row: int, title: str, value_label: QLabel) -> None:
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label.setObjectName("metricValue")
        layout.addWidget(title_label, 0, row)
        layout.addWidget(value_label, 1, row)

    def _connect_signals(self) -> None:
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.fullscreen_play_pause_button.clicked.connect(self.toggle_playback)
        self.prev_segment_button.clicked.connect(self.jump_to_previous_segment)
        self.next_segment_button.clicked.connect(self.jump_to_next_segment)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_exit_button.clicked.connect(self.exit_fullscreen)
        self.content_only_checkbox.toggled.connect(
            lambda checked: self.on_content_only_changed(checked, self.content_only_checkbox)
        )
        self.fullscreen_content_only_checkbox.toggled.connect(
            lambda checked: self.on_content_only_changed(checked, self.fullscreen_content_only_checkbox)
        )
        self.volume_slider.valueChanged.connect(lambda value: self.on_volume_changed(value, self.volume_slider))
        self.fullscreen_volume_slider.valueChanged.connect(
            lambda value: self.on_volume_changed(value, self.fullscreen_volume_slider)
        )
        self.speed_combo.currentIndexChanged.connect(lambda _index: self.on_speed_changed(self.speed_combo))
        self.fullscreen_speed_combo.currentIndexChanged.connect(
            lambda _index: self.on_speed_changed(self.fullscreen_speed_combo)
        )

        self.position_slider.sliderPressed.connect(lambda: self.on_slider_pressed(self.position_slider))
        self.position_slider.sliderReleased.connect(lambda: self.on_slider_released(self.position_slider))
        self.position_slider.sliderMoved.connect(lambda value: self.on_slider_moved(value, self.position_slider))
        self.position_slider.positionClicked.connect(lambda value: self.on_slider_moved(value, self.position_slider))
        self.position_slider.positionDragged.connect(lambda value: self.on_slider_moved(value, self.position_slider))
        self.fullscreen_position_slider.sliderPressed.connect(
            lambda: self.on_slider_pressed(self.fullscreen_position_slider)
        )
        self.fullscreen_position_slider.sliderReleased.connect(
            lambda: self.on_slider_released(self.fullscreen_position_slider)
        )
        self.fullscreen_position_slider.sliderMoved.connect(
            lambda value: self.on_slider_moved(value, self.fullscreen_position_slider)
        )
        self.fullscreen_position_slider.positionClicked.connect(
            lambda value: self.on_slider_moved(value, self.fullscreen_position_slider)
        )
        self.fullscreen_position_slider.positionDragged.connect(
            lambda value: self.on_slider_moved(value, self.fullscreen_position_slider)
        )

        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.player.errorOccurred.connect(self.on_player_error)
        self.video_widget.fullScreenChanged.connect(self.on_fullscreen_changed)
        self.video_widget.toggleFullScreenRequested.connect(self.toggle_fullscreen)
        self.video_widget.exitFullScreenRequested.connect(self.exit_fullscreen)
        self.video_widget.togglePlaybackRequested.connect(self.toggle_playback)
        self.video_widget.pointerMoved.connect(self.on_video_pointer_moved)
        self.video_widget.widgetResized.connect(self.on_video_widget_resized)
        self.fullscreen_cursor_poll_timer.timeout.connect(self.poll_fullscreen_cursor)
        self.processing_animation_timer.timeout.connect(self.processing_spinner.advance)

        self.timeline_widget.segmentSelected.connect(self.select_segment_by_id)
        self.timeline_widget.segmentActivated.connect(self.on_timeline_segment_activated)

        self.segment_table.itemSelectionChanged.connect(self.on_segment_table_selection_changed)
        self.segment_table.cellDoubleClicked.connect(self.on_segment_table_double_clicked)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f1e6;
            }
            QGroupBox {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(24, 33, 47, 0.12);
                border-radius: 18px;
                font-weight: 700;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QVideoWidget {
                background: #05070a;
                border-radius: 18px;
            }
            QVideoWidget[fullscreenActive="true"] {
                background: #000000;
                border-radius: 0px;
            }
            QPushButton, QToolButton {
                background: #f8fafc;
                border: 1px solid rgba(24, 33, 47, 0.12);
                border-radius: 12px;
                padding: 8px 14px;
                font-size: 13px;
                font-weight: 400;
            }
            QPushButton:hover, QToolButton:hover {
                background: #eef2f6;
            }
            QCheckBox#inlineCheck, QComboBox#inlineCombo, QLabel#inlineLabel {
                font-size: 13px;
                font-weight: 400;
                color: #18212f;
            }
            QCheckBox#inlineCheck:disabled, QComboBox#inlineCombo:disabled {
                color: #9aa5b1;
            }
            QToolButton#mediaButton {
                background: #f8fbff;
                color: #102a43;
                border: 1px solid #a8bed6;
                border-radius: 18px;
                min-width: 38px;
                max-width: 38px;
                min-height: 38px;
                max-height: 38px;
                padding: 0;
            }
            QToolButton#mediaButton:hover {
                background: #d9eef3;
                border: 1px solid #7aa8b7;
            }
            QToolButton#mediaButton:pressed {
                background: #bfe0e8;
                border: 1px solid #4e7b88;
            }
            QFrame#fullscreenOverlay {
                background: rgba(10, 14, 22, 220);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 18px;
            }
            QFrame#processingOverlay {
                background: rgba(10, 14, 22, 210);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 18px;
            }
            QLabel#processingOverlayLabel {
                color: #f8fafc;
                font-size: 15px;
                font-weight: 700;
            }
            QToolButton#overlayMediaButton {
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 18px;
                min-width: 38px;
                max-width: 38px;
                min-height: 38px;
                max-height: 38px;
                padding: 0;
            }
            QToolButton#overlayMediaButton:hover {
                background: rgba(255, 255, 255, 0.24);
            }
            QToolButton#overlayMediaButton:pressed {
                background: rgba(255, 255, 255, 0.32);
            }
            QLabel#overlayTimeLabel {
                color: #f8fafc;
                font-weight: 600;
            }
            QLabel#overlayInlineLabel {
                color: #f8fafc;
                font-size: 13px;
                font-weight: 400;
            }
            QLabel#overlayInlineLabel[processingActive="true"] {
                color: rgba(248, 250, 252, 0.42);
            }
            QLabel#overlayInlineLabel:disabled {
                color: rgba(248, 250, 252, 0.42);
            }
            QToolButton#overlayToggleButton {
                background: transparent;
                color: #f8fafc;
                border: none;
                padding: 2px 4px;
                font-size: 13px;
                font-weight: 400;
                text-align: left;
            }
            QToolButton#overlayToggleButton:hover {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 8px;
            }
            QToolButton#overlayToggleButton:pressed {
                background: rgba(255, 255, 255, 0.14);
            }
            QToolButton#overlayToggleButton:disabled {
                color: rgba(248, 250, 252, 0.42);
            }
            QComboBox#overlayCombo {
                background: rgba(255, 255, 255, 0.14);
                color: #f8fafc;
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 10px;
                padding: 6px 8px;
                min-width: 78px;
                font-size: 13px;
                font-weight: 400;
            }
            QComboBox#overlayCombo:disabled {
                color: rgba(248, 250, 252, 0.42);
                border: 1px solid rgba(255, 255, 255, 0.10);
            }
            QComboBox#overlayCombo:hover {
                background: rgba(255, 255, 255, 0.22);
                border: 1px solid rgba(255, 255, 255, 0.28);
            }
            QComboBox#overlayCombo::drop-down {
                border: none;
                width: 22px;
            }
            QComboBox#overlayCombo::down-arrow {
                image: none;
                width: 0;
                height: 0;
            }
            QListView#overlayComboPopup {
                background: rgba(15, 22, 34, 244);
                color: #f8fafc;
                border: 1px solid rgba(255, 255, 255, 0.20);
                border-radius: 10px;
                padding: 6px;
                outline: none;
            }
            QListView#overlayComboPopup::item {
                padding: 8px 10px;
                border-radius: 8px;
                min-height: 20px;
            }
            QListView#overlayComboPopup::item:selected {
                background: rgba(77, 171, 247, 0.35);
                color: #ffffff;
            }
            QListView#overlayComboPopup::item:hover {
                background: rgba(255, 255, 255, 0.12);
                color: #ffffff;
            }
            QSlider#overlayVolumeSlider::groove:horizontal,
            QSlider#fullscreenPositionSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.18);
                border-radius: 3px;
            }
            QSlider#overlayVolumeSlider::sub-page:horizontal {
                background: #4dabf7;
                border-radius: 3px;
            }
            QSlider#overlayVolumeSlider[processingActive="true"]::sub-page:horizontal {
                background: #93d5ff;
                border-radius: 3px;
            }
            QSlider#fullscreenPositionSlider::sub-page:horizontal {
                background: #ff922b;
                border-radius: 3px;
            }
            QSlider#overlayVolumeSlider::handle:horizontal,
            QSlider#fullscreenPositionSlider::handle:horizontal {
                background: #f8fafc;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider#overlayVolumeSlider::handle:horizontal {
                background: #4dabf7;
            }
            QSlider#overlayVolumeSlider[processingActive="true"]::handle:horizontal {
                background: #93d5ff;
            }
            QComboBox, QTableWidget {
                background: #ffffff;
                border: 1px solid rgba(24, 33, 47, 0.14);
                border-radius: 10px;
                padding: 6px;
            }
            QSlider#volumeSlider::groove:horizontal {
                height: 6px;
                background: #d8d2c5;
                border-radius: 3px;
            }
            QSlider#volumeSlider::sub-page:horizontal {
                background: #1f6f8b;
                border-radius: 3px;
            }
            QSlider#volumeSlider[processingActive="true"]::sub-page:horizontal {
                background: #9ed3f5;
                border-radius: 3px;
            }
            QSlider#volumeSlider::handle:horizontal {
                background: #1f6f8b;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider#volumeSlider[processingActive="true"]::handle:horizontal {
                background: #9ed3f5;
            }
            QLabel#inlineLabel {
                color: #18212f;
            }
            QLabel#inlineLabel[processingActive="true"] {
                color: #9aa5b1;
            }
            QLabel#inlineLabel:disabled {
                color: #9aa5b1;
            }
            QLabel#metricTitle {
                color: #5b6878;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#metricValue {
                color: #18212f;
                font-size: 24px;
                font-weight: 800;
            }
            QLabel#metricLabel, QLabel#sourceLabel {
                color: #445265;
                font-weight: 600;
            }
            QHeaderView::section {
                background: #f0ece2;
                border: none;
                border-bottom: 1px solid rgba(24, 33, 47, 0.12);
                padding: 8px;
                font-weight: 700;
            }
            QTableWidget {
                gridline-color: rgba(24, 33, 47, 0.08);
                alternate-background-color: #faf8f2;
                selection-background-color: #ffe08a;
                selection-color: #18212f;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #d8d2c5;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1f6f8b;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            """
        )
        self.fullscreen_controls.setStyleSheet(self.styleSheet())
        self.processing_overlay.setStyleSheet(self.styleSheet())

    def _load_initial_state(self) -> None:
        self.refresh_ui()
        self.statusBar().showMessage("Ready", 2500)

    def refresh_ui(self) -> None:
        self.refresh_sources()
        self.refresh_metrics()
        self.refresh_timeline()
        self.refresh_position_label()
        self.refresh_segment_table()
        self.refresh_navigation_buttons()
        self.refresh_fullscreen_availability()
        self.refresh_volume_availability()

    def refresh_sources(self) -> None:
        self.video_source_label.setText(f"Video: {self.current_video_path.name if self.current_video_path else 'none'}")

    def has_loaded_video(self) -> bool:
        return not self.player.source().isEmpty()

    def refresh_fullscreen_availability(self) -> None:
        fullscreen_enabled = self.has_loaded_video() and not self.is_processing_active()
        self.fullscreen_button.setEnabled(fullscreen_enabled)

    def refresh_volume_availability(self) -> None:
        processing_active = self.is_processing_active()
        for label in (self.volume_label, self.fullscreen_volume_label):
            label.setEnabled(True)
            label.setProperty("processingActive", processing_active)
            label.style().unpolish(label)
            label.style().polish(label)
            label.update()
        for slider in (self.volume_slider, self.fullscreen_volume_slider):
            slider.setEnabled(True)
            slider.setProperty("processingActive", processing_active)
            slider.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, processing_active)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus if processing_active else Qt.FocusPolicy.StrongFocus)
            if processing_active:
                slider.clearFocus()
            slider.style().unpolish(slider)
            slider.style().polish(slider)
            slider.update()

    def set_processing_visible(self, visible: bool) -> None:
        self.processing_overlay_visible = visible
        if visible:
            self.player.pause()
            if not self.processing_animation_timer.isActive():
                self.processing_animation_timer.start()
            self.update_processing_overlay_geometry()
            self.processing_overlay.show()
            self.processing_overlay.raise_()
        else:
            self.processing_animation_timer.stop()
            self.processing_overlay.hide()
            self.processing_spinner.reset()
        self.open_video_action.setEnabled(not visible)
        for shortcut in (
            self.window_fullscreen_shortcut,
            self.window_exit_shortcut,
            self.fullscreen_toggle_shortcut,
            self.fullscreen_exit_shortcut,
            self.overlay_exit_shortcut,
            self.overlay_fullscreen_shortcut,
        ):
            shortcut.setEnabled(not visible)
        for widget in (
            self.play_pause_button,
            self.prev_segment_button,
            self.next_segment_button,
            self.fullscreen_button,
            self.timeline_widget,
            self.position_slider,
            self.fullscreen_play_pause_button,
            self.fullscreen_position_slider,
            self.fullscreen_exit_button,
            self.segment_table,
            self.content_only_checkbox,
            self.fullscreen_content_only_checkbox,
            self.speed_label,
            self.fullscreen_speed_label,
            self.speed_combo,
            self.fullscreen_speed_combo,
        ):
            widget.setEnabled(not visible)
        self.refresh_volume_availability()
        if not visible:
            self.refresh_navigation_buttons()
            self.refresh_fullscreen_availability()

    def is_processing_active(self) -> bool:
        return self.processing_thread is not None or self.processing_overlay_visible

    def is_fullscreen_active(self) -> bool:
        return self.video_widget.isFullScreen()

    def show_processing_overlay(self) -> None:
        self.processing_label.setText("Processing...")
        self.set_processing_visible(True)
        QApplication.processEvents()

    def hide_processing_overlay(self) -> None:
        self.set_processing_visible(False)

    def refresh_metrics(self) -> None:
        content_seconds = sum(segment.duration for segment in self.segments if segment.kind == "content")
        non_content_seconds = sum(segment.duration for segment in self.segments if segment.kind != "content")
        self.total_segments_value.setText(str(len(self.segments)))
        self.content_duration_value.setText(format_time(content_seconds))
        self.non_content_duration_value.setText(format_time(non_content_seconds))

    def refresh_timeline(self) -> None:
        self.timeline_widget.set_state(
            segments=self.segments,
            duration_seconds=self.current_duration_seconds(),
            position_seconds=self.player.position() / 1000.0,
            selected_segment_id=self.selected_segment_id,
            active_segment_id=self.active_segment_id,
        )

    def refresh_position_label(self) -> None:
        current_seconds = self.player.position() / 1000.0
        total_seconds = self.current_duration_seconds()
        self._update_time_labels(current_seconds, total_seconds)
        active_segment = self.segment_by_id(self.active_segment_id)
        if active_segment:
            self.segment_badge_label.setText(
                f"{active_segment.label_name} | {format_time(active_segment.start)} - {format_time(active_segment.end)}"
            )
        else:
            self.segment_badge_label.setText("No active segment")

    def refresh_segment_table(self) -> None:
        with QSignalBlocker(self.segment_table):
            self.segment_table.setRowCount(len(self.segments))
            for row, segment in enumerate(self.segments):
                values = [
                    segment.label_name,
                    f"{format_time(segment.start)} - {format_time(segment.end)}",
                ]
                for column, text in enumerate(values):
                    item = self.segment_table.item(row, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self.segment_table.setItem(row, column, item)
                    item.setText(text)
                    item.setData(Qt.ItemDataRole.UserRole, segment.identifier)
                    if segment.identifier == self.active_segment_id:
                        item.setBackground(QColor("#d9ebff"))
                    elif segment.identifier == self.selected_segment_id:
                        item.setBackground(QColor("#ffe08a"))
                    else:
                        item.setBackground(QColor("#ffffff") if row % 2 == 0 else QColor("#faf8f2"))

            selected_row = self.index_for_segment_id(self.selected_segment_id)
            if selected_row != -1:
                self.segment_table.selectRow(selected_row)
            else:
                self.segment_table.clearSelection()

    def current_duration_seconds(self) -> float:
        player_duration = self.player.duration() / 1000.0
        if player_duration > 0:
            return player_duration
        return max((segment.end for segment in self.segments), default=0.0)

    def open_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(self.current_video_path.parent if self.current_video_path else Path.cwd()),
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All Files (*)",
        )
        if filename:
            self.load_video_path(Path(filename))

    def clear_current_segments(self) -> None:
        self.segments = []
        self.selected_segment_id = None
        self.active_segment_id = None
        self.user_selected_segment = False
        self.refresh_ui()

    def apply_segments(
        self,
        *,
        segments: list[Segment],
        status_message: str | None = None,
    ) -> None:
        self.segments = segments
        self.selected_segment_id = segments[0].identifier if segments else None
        self.active_segment_id = None
        self.user_selected_segment = False
        self._refresh_position_slider_range()
        self.refresh_ui()
        if status_message:
            self.statusBar().showMessage(status_message, 2500)

    def process_current_video(self) -> None:
        if self.current_video_path is None or self.processing_thread is not None:
            return
        self.show_processing_overlay()
        self.processing_thread = QThread(self)
        self.processing_worker = SegmentationWorker(self.current_video_path)
        self.processing_worker.moveToThread(self.processing_thread)
        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.finished.connect(self.on_processing_finished)
        self.processing_worker.failed.connect(self.on_processing_failed)
        self.processing_worker.finished.connect(self.processing_thread.quit)
        self.processing_worker.failed.connect(self.processing_thread.quit)
        self.processing_thread.finished.connect(self.cleanup_processing_thread)
        self.processing_thread.start()

    def load_video_path(self, path: Path) -> None:
        if self.processing_thread is not None:
            return
        self.current_video_path = path.resolve()
        self.show_processing_overlay()
        self.player.setSource(QUrl.fromLocalFile(str(self.current_video_path)))
        self.clear_current_segments()
        QTimer.singleShot(0, self.process_current_video)

    def on_processing_finished(self, segments: object) -> None:
        if not isinstance(segments, list):
            self.on_processing_failed("Segmentation returned an invalid result.")
            return
        self.apply_segments(
            segments=segments,
            status_message=f"Processing complete: {len(segments)} segments",
        )
        self.hide_processing_overlay()

    def on_processing_failed(self, message: str) -> None:
        self.clear_current_segments()
        self.hide_processing_overlay()
        self.show_error(f"Video processing failed: {message}")

    def cleanup_processing_thread(self) -> None:
        if self.processing_worker is not None:
            self.processing_worker.deleteLater()
        if self.processing_thread is not None:
            self.processing_thread.deleteLater()
        self.processing_worker = None
        self.processing_thread = None
        self.refresh_navigation_buttons()
        self.refresh_fullscreen_availability()
        self.refresh_volume_availability()

    def toggle_playback(self) -> None:
        if self.is_processing_active():
            return
        if self.player.source().isEmpty():
            self.show_error("Load a video before using playback controls.")
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def jump_to_previous_segment(self) -> None:
        if not self.segments:
            return
        target = self.previous_navigation_target(self.player.position() / 1000.0)
        if target is not None:
            self.jump_to_segment_start(target)

    def jump_to_next_segment(self) -> None:
        if not self.segments:
            return
        target = self.next_navigation_target(self.player.position() / 1000.0)
        if target is not None:
            self.jump_to_segment_start(target)

    def toggle_fullscreen(self) -> None:
        if self.is_processing_active():
            return
        if not self.has_loaded_video():
            return
        self.video_widget.setFullScreen(not self.video_widget.isFullScreen())

    def exit_fullscreen(self) -> None:
        if self.is_processing_active():
            return
        if self.video_widget.isFullScreen():
            self.video_widget.setFullScreen(False)

    def on_content_only_toggled(self, checked: bool) -> None:
        self._refresh_position_slider_range()
        if checked:
            self.statusBar().showMessage("Content-only mode enabled", 3000)
            self.enforce_content_only(
                self.player.position() / 1000.0,
                keep_playing=self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState,
            )
            bounded_position = self.clamp_seek_milliseconds(self.player.position())
            if bounded_position != self.player.position():
                self.player.pause()
                self.seek_to_milliseconds(bounded_position)
        else:
            self.statusBar().showMessage("Content-only mode disabled", 3000)
        self.refresh_navigation_buttons()

    def on_content_only_changed(self, checked: bool, source: Any) -> None:
        target = self.fullscreen_content_only_checkbox if source is self.content_only_checkbox else self.content_only_checkbox
        with QSignalBlocker(target):
            target.setChecked(checked)
        self._update_content_only_checkbox_icons(checked)
        self.on_content_only_toggled(checked)

    def on_volume_changed(self, value: int, source: QSlider | None = None) -> None:
        if self.is_processing_active():
            return
        for slider in (self.volume_slider, self.fullscreen_volume_slider):
            if slider is source:
                continue
            with QSignalBlocker(slider):
                slider.setValue(value)
        self.audio_output.setVolume(max(0.0, min(1.0, value / 100.0)))

    def on_speed_changed(self, source: QComboBox | None = None) -> None:
        combo = source or self.speed_combo
        target = self.fullscreen_speed_combo if combo is self.speed_combo else self.speed_combo
        with QSignalBlocker(target):
            target.setCurrentIndex(combo.currentIndex())
        self.player.setPlaybackRate(float(combo.currentData()))

    def on_fullscreen_popup_shown(self) -> None:
        self.fullscreen_popup_open = True
        self._mark_fullscreen_activity()

    def on_fullscreen_popup_hidden(self) -> None:
        self.fullscreen_popup_open = False
        self.last_fullscreen_activity_at = monotonic()

    def on_slider_pressed(self, slider: PositionSlider | None = None) -> None:
        self.slider_is_active = True
        self.active_position_slider = slider
        if self.is_fullscreen_active():
            self._mark_fullscreen_activity()

    def on_slider_released(self, slider: PositionSlider | None = None) -> None:
        target_slider = slider or self.active_position_slider or self.position_slider
        self.slider_is_active = False
        self.active_position_slider = None
        bounded_value = self.constrain_user_scrub_milliseconds(target_slider.value())
        self.seek_to_milliseconds(bounded_value)
        self._sync_position_sliders(bounded_value, exclude=target_slider)
        if self.is_fullscreen_active():
            self.last_fullscreen_activity_at = monotonic()

    def on_slider_moved(self, value: int, slider: PositionSlider | None = None) -> None:
        bounded_value = self.constrain_user_scrub_milliseconds(value)
        self.seek_to_milliseconds(bounded_value)
        self._sync_position_sliders(bounded_value)
        self._update_time_labels(bounded_value / 1000.0, self.current_duration_seconds())
        if self.is_fullscreen_active():
            self.last_fullscreen_activity_at = monotonic()

    def seek_to_milliseconds(self, value: int) -> None:
        self.player.setPosition(self.clamp_seek_milliseconds(value))

    def seek_to_seconds(self, value: float) -> None:
        self.seek_to_milliseconds(int(value * 1000))

    def on_player_position_changed(self, position_ms: int) -> None:
        if not self.slider_is_active:
            self._sync_position_sliders(position_ms)

        active_segment = self.segment_for_time(position_ms / 1000.0)
        self.active_segment_id = active_segment.identifier if active_segment else None

        if active_segment and not self.user_selected_segment:
            self.selected_segment_id = active_segment.identifier

        self.refresh_position_label()
        self.refresh_timeline()
        self.refresh_segment_table()
        self.refresh_navigation_buttons()

        if self.content_only_checkbox.isChecked():
            self.enforce_content_only(
                position_ms / 1000.0,
                keep_playing=self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState,
            )

    def on_player_duration_changed(self, duration_ms: int) -> None:
        self._refresh_position_slider_range()
        self.refresh_position_label()
        self.refresh_timeline()

    def on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self._update_playback_button_icon(state)

    def on_player_error(self, error: QMediaPlayer.Error, error_string: str) -> None:
        if error != QMediaPlayer.Error.NoError:
            self.show_error(error_string or "Media playback failed.")

    def on_fullscreen_changed(self, is_fullscreen: bool) -> None:
        self.video_widget.setProperty("fullscreenActive", is_fullscreen)
        self.video_widget.style().unpolish(self.video_widget)
        self.video_widget.style().polish(self.video_widget)
        self.video_widget.update()
        self._update_fullscreen_button_icon(is_fullscreen)
        if is_fullscreen:
            self.video_widget.setFocus()
            self.last_fullscreen_cursor_pos = QCursor.pos()
            self.last_fullscreen_activity_at = monotonic()
            self._set_fullscreen_cursor_hidden(False)
            self.fullscreen_cursor_poll_timer.start()
            QTimer.singleShot(0, self.update_fullscreen_overlay_geometry)
            QTimer.singleShot(0, self.show_fullscreen_controls)
        else:
            self.fullscreen_cursor_poll_timer.stop()
            self.fullscreen_controls.hide()
            self.fullscreen_popup_open = False
            self._set_fullscreen_cursor_hidden(False)

    def on_video_pointer_moved(self, point: QPoint) -> None:
        del point
        if not self.is_fullscreen_active():
            return
        self._mark_fullscreen_activity()

    def poll_fullscreen_cursor(self) -> None:
        if not self.is_fullscreen_active():
            self.fullscreen_cursor_poll_timer.stop()
            return

        current_cursor_pos = QCursor.pos()
        top_left = self.video_widget.mapToGlobal(QPoint(0, 0))
        within_video = (
            top_left.x() <= current_cursor_pos.x() <= top_left.x() + self.video_widget.width()
            and top_left.y() <= current_cursor_pos.y() <= top_left.y() + self.video_widget.height()
        )
        moved_distance = abs(current_cursor_pos.x() - self.last_fullscreen_cursor_pos.x()) + abs(
            current_cursor_pos.y() - self.last_fullscreen_cursor_pos.y()
        )
        if moved_distance >= 1:
            self.last_fullscreen_cursor_pos = current_cursor_pos
            if within_video:
                self._mark_fullscreen_activity()

        if (
            within_video
            and self.fullscreen_controls.isVisible()
            and not self.slider_is_active
            and not self.is_fullscreen_popup_visible()
            and monotonic() - self.last_fullscreen_activity_at >= self.fullscreen_idle_hide_seconds
        ):
            self.fullscreen_controls.hide()
            self._set_fullscreen_cursor_hidden(True)

    def segment_for_time(self, time_seconds: float) -> Segment | None:
        for segment in self.segments:
            if segment.start - EPSILON_SECONDS <= time_seconds < segment.end - (EPSILON_SECONDS / 2):
                return segment
        return None

    def navigation_segments(self) -> list[Segment]:
        if self.content_only_checkbox.isChecked():
            content_segments = [segment for segment in self.segments if segment.kind == "content"]
            if content_segments:
                return content_segments
        return self.segments

    def navigation_anchor_index(self, time_seconds: float) -> int | None:
        navigation_segments = self.navigation_segments()
        if not navigation_segments:
            return None

        if time_seconds < navigation_segments[0].start - EPSILON_SECONDS:
            return -1

        for index, segment in enumerate(navigation_segments):
            if segment.start - EPSILON_SECONDS <= time_seconds < segment.end - (EPSILON_SECONDS / 2):
                return index

        previous_index = -1
        for index, segment in enumerate(navigation_segments):
            if segment.start <= time_seconds + EPSILON_SECONDS:
                previous_index = index
            else:
                break
        return previous_index

    def can_navigate_previous(self, time_seconds: float) -> bool:
        anchor_index = self.navigation_anchor_index(time_seconds)
        return anchor_index is not None and anchor_index > 0

    def can_navigate_next(self, time_seconds: float) -> bool:
        navigation_segments = self.navigation_segments()
        anchor_index = self.navigation_anchor_index(time_seconds)
        if anchor_index is None:
            return False
        if anchor_index < 0:
            return bool(navigation_segments)
        return anchor_index < len(navigation_segments) - 1

    def refresh_navigation_buttons(self) -> None:
        if self.is_processing_active():
            prev_enabled = False
            next_enabled = False
        else:
            current_seconds = self.player.position() / 1000.0
            prev_enabled = self.can_navigate_previous(current_seconds)
            next_enabled = self.can_navigate_next(current_seconds)
        self.prev_segment_button.setEnabled(prev_enabled)
        self.next_segment_button.setEnabled(next_enabled)

    def previous_navigation_target(self, time_seconds: float) -> Segment | None:
        navigation_segments = self.navigation_segments()
        if not navigation_segments:
            return None

        for index, segment in enumerate(navigation_segments):
            if segment.start - EPSILON_SECONDS <= time_seconds < segment.end - (EPSILON_SECONDS / 2):
                return navigation_segments[index - 1] if index > 0 else None

        previous_segment: Segment | None = None
        for segment in navigation_segments:
            if segment.start < time_seconds - EPSILON_SECONDS:
                previous_segment = segment
            else:
                break
        return previous_segment

    def next_navigation_target(self, time_seconds: float) -> Segment | None:
        navigation_segments = self.navigation_segments()
        if not navigation_segments:
            return None

        for index, segment in enumerate(navigation_segments):
            if segment.start - EPSILON_SECONDS <= time_seconds < segment.end - (EPSILON_SECONDS / 2):
                return navigation_segments[index + 1] if index < len(navigation_segments) - 1 else None

        for segment in navigation_segments:
            if segment.start > time_seconds + EPSILON_SECONDS:
                return segment
        return None

    def jump_to_segment_start(self, segment: Segment) -> None:
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.select_segment_by_id(segment.identifier, preserve_manual=True)
        self.seek_to_seconds(segment.start + EPSILON_SECONDS)
        if was_playing and not self.is_processing_active() and not self.player.source().isEmpty():
            self.player.play()

    def next_content_segment_after(self, time_seconds: float) -> Segment | None:
        for segment in self.segments:
            if segment.kind == "content" and segment.end > time_seconds + EPSILON_SECONDS:
                return segment
        return None

    def enforce_content_only(self, current_time_seconds: float, *, keep_playing: bool) -> None:
        active_segment = self.segment_for_time(current_time_seconds)
        if active_segment is None or active_segment.kind == "content":
            return

        next_content = self.next_content_segment_after(active_segment.end)
        if next_content is None:
            self.seek_to_seconds(active_segment.start)
            self.player.pause()
            return

        self.seek_to_seconds(next_content.start + EPSILON_SECONDS)
        self.select_segment_by_id(next_content.identifier, preserve_manual=False)
        if keep_playing and not self.player.source().isEmpty() and not self.is_processing_active():
            self.player.play()

    def segment_by_id(self, identifier: str | None) -> Segment | None:
        if not identifier:
            return None
        for segment in self.segments:
            if segment.identifier == identifier:
                return segment
        return None

    def index_for_segment_id(self, identifier: str | None) -> int:
        if not identifier:
            return -1
        for index, segment in enumerate(self.segments):
            if segment.identifier == identifier:
                return index
        return -1

    def select_segment_by_id(self, identifier: str, preserve_manual: bool = True) -> None:
        if self.segment_by_id(identifier) is None:
            return
        self.selected_segment_id = identifier
        self.user_selected_segment = preserve_manual
        self.refresh_timeline()
        self.refresh_segment_table()

    def on_segment_table_selection_changed(self) -> None:
        selected_items = self.segment_table.selectedItems()
        if not selected_items:
            return
        identifier = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(identifier, str):
            self.selected_segment_id = identifier
            self.user_selected_segment = True
            self.refresh_timeline()
            self.refresh_segment_table()

    def on_segment_table_double_clicked(self, row: int, column: int) -> None:
        del column
        if self.is_processing_active():
            return
        if not (0 <= row < len(self.segments)):
            return
        segment = self.segments[row]
        self.jump_to_segment_start(segment)

    def on_timeline_segment_activated(self, identifier: str) -> None:
        if self.is_processing_active():
            return
        segment = self.segment_by_id(identifier)
        if segment is None:
            return
        self.jump_to_segment_start(segment)

    def _belongs_to_player_window(self, widget: QWidget | None) -> bool:
        if widget is None:
            return False
        return (
            widget is self
            or self.isAncestorOf(widget)
            or widget is self.video_widget
            or widget is self.fullscreen_controls
            or self.video_widget.isAncestorOf(widget)
            or self.fullscreen_controls.isAncestorOf(widget)
        )

    def _should_handle_player_hotkey(self, watched_widget: QWidget | None) -> bool:
        return (
            self.isActiveWindow()
            or self.is_fullscreen_active()
            or self.fullscreen_controls.isVisible()
            or self._belongs_to_player_window(watched_widget)
        )

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress) and hasattr(event, "key"):
            if QApplication.activeModalWidget() is None:
                watched_widget = watched if isinstance(watched, QWidget) else QApplication.focusWidget()
                if self._should_handle_player_hotkey(watched_widget):
                    key = event.key()
                    if key == Qt.Key.Key_Space and event.type() == QEvent.Type.KeyPress:
                        self.toggle_playback()
                        event.accept()
                        return True
                    if self.is_processing_active() and key in (Qt.Key.Key_F, Qt.Key.Key_Escape):
                        event.accept()
                        return True
        if watched in self.fullscreen_overlay_watchers:
            if event.type() in (QEvent.Type.Enter, QEvent.Type.MouseMove, QEvent.Type.Show):
                self._mark_fullscreen_activity()
        return super().eventFilter(watched, event)

    def show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        QMessageBox.warning(self, "Player Error", message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.processing_thread is not None:
            self.statusBar().showMessage("Processing is still running. Please wait for it to finish.", 4000)
            event.ignore()
            return
        if self.is_fullscreen_active():
            self.exit_fullscreen()
        super().closeEvent(event)

    def _sync_position_sliders(self, value: int, exclude: PositionSlider | None = None) -> None:
        bounded_value = self.clamp_seek_milliseconds(value)
        for slider in (self.position_slider, self.fullscreen_position_slider):
            if slider is exclude:
                continue
            with QSignalBlocker(slider):
                slider.setValue(bounded_value)

    def _set_position_slider_range(self, maximum: int) -> None:
        bounded_maximum = max(0, maximum)
        for slider in (self.position_slider, self.fullscreen_position_slider):
            with QSignalBlocker(slider):
                slider.setRange(0, bounded_maximum)

    def last_content_end_seconds(self) -> float:
        return max((segment.end for segment in self.segments if segment.kind == "content"), default=0.0)

    def maximum_allowed_seek_seconds(self) -> float:
        total_seconds = self.current_duration_seconds()
        if self.content_only_checkbox.isChecked():
            last_content_end = self.last_content_end_seconds()
            if last_content_end > 0:
                return min(last_content_end, total_seconds)
        return total_seconds

    def maximum_seek_milliseconds(self) -> int:
        return max(0, int(self.current_duration_seconds() * 1000))

    def maximum_allowed_seek_milliseconds(self) -> int:
        return max(0, int(self.maximum_allowed_seek_seconds() * 1000))

    def clamp_seek_milliseconds(self, value: int) -> int:
        return max(0, min(int(value), self.maximum_allowed_seek_milliseconds()))

    def constrain_user_scrub_milliseconds(self, value: int) -> int:
        bounded_value = self.clamp_seek_milliseconds(value)
        if not self.content_only_checkbox.isChecked():
            return bounded_value

        requested_seconds = bounded_value / 1000.0
        active_segment = self.segment_for_time(requested_seconds)
        if active_segment is not None and active_segment.kind == "content":
            return bounded_value

        reference_milliseconds = self.player.position()
        moving_forward = bounded_value >= reference_milliseconds
        previous_content: Segment | None = None
        next_content: Segment | None = None

        for segment in self.segments:
            if segment.kind != "content":
                continue
            if segment.end <= requested_seconds + EPSILON_SECONDS:
                previous_content = segment
            if next_content is None and segment.start >= requested_seconds - EPSILON_SECONDS:
                next_content = segment

        if moving_forward:
            if previous_content is not None:
                return int(previous_content.end * 1000)
            if next_content is not None:
                return int(next_content.start * 1000)
        else:
            if next_content is not None:
                return int(next_content.start * 1000)
            if previous_content is not None:
                return int(previous_content.end * 1000)

        return bounded_value

    def _refresh_position_slider_range(self) -> None:
        self._set_position_slider_range(self.maximum_seek_milliseconds())
        interaction_maximum = (
            self.maximum_allowed_seek_milliseconds() if self.content_only_checkbox.isChecked() else None
        )
        for slider in (self.position_slider, self.fullscreen_position_slider):
            with QSignalBlocker(slider):
                slider.setInteractionMaximum(interaction_maximum)
                if interaction_maximum is not None and slider.value() > interaction_maximum:
                    slider.setValue(interaction_maximum)

    def _update_time_labels(self, current_seconds: float, total_seconds: float) -> None:
        text = f"{format_time(current_seconds)} / {format_time(total_seconds)}"
        self.time_label.setText(text)
        self.fullscreen_time_label.setText(text)

    def on_video_widget_resized(self) -> None:
        self.update_fullscreen_overlay_geometry()
        self.update_processing_overlay_geometry()

    def update_processing_overlay_geometry(self) -> None:
        if not hasattr(self, "processing_overlay"):
            return
        top_left = self.video_widget.mapToGlobal(QPoint(0, 0))
        overlay_width = min(260, max(180, self.video_widget.width() - 40))
        overlay_height = 110
        x = top_left.x() + max(12, (self.video_widget.width() - overlay_width) // 2)
        y = top_left.y() + max(12, (self.video_widget.height() - overlay_height) // 2)
        self.processing_overlay.setGeometry(x, y, overlay_width, overlay_height)
        if self.processing_overlay_visible:
            self.processing_overlay.raise_()

    def update_fullscreen_overlay_geometry(self) -> None:
        if not hasattr(self, "fullscreen_controls"):
            return
        top_left = self.video_widget.mapToGlobal(QPoint(0, 0))
        available_width = max(240, self.video_widget.width() - 36)
        overlay_width = min(max(420, available_width), max(240, self.video_widget.width() - 18))
        overlay_height = 104
        x = top_left.x() + max(9, (self.video_widget.width() - overlay_width) // 2)
        y = top_left.y() + max(18, self.video_widget.height() - overlay_height - 18)
        self.fullscreen_controls.setGeometry(x, y, overlay_width, overlay_height)
        if not self.is_fullscreen_popup_visible():
            self.fullscreen_controls.raise_()

    def show_fullscreen_controls(self) -> None:
        if not self.is_fullscreen_active():
            return
        self._set_fullscreen_cursor_hidden(False)
        self.update_fullscreen_overlay_geometry()
        self.fullscreen_controls.show()
        if not self.is_fullscreen_popup_visible():
            self.fullscreen_controls.raise_()

    def _update_content_only_checkbox_icons(self, checked: bool) -> None:
        self.fullscreen_content_only_checkbox.setIcon(self._draw_checkbox_icon(checked))

    def _mark_fullscreen_activity(self) -> None:
        self.last_fullscreen_activity_at = monotonic()
        self.show_fullscreen_controls()

    def is_fullscreen_popup_visible(self) -> bool:
        return self.fullscreen_popup_open

    def _set_fullscreen_cursor_hidden(self, hidden: bool) -> None:
        if self.fullscreen_cursor_hidden == hidden:
            return
        cursor_shape = Qt.CursorShape.BlankCursor if hidden else Qt.CursorShape.ArrowCursor
        for widget in (self.video_widget, self.fullscreen_controls):
            widget.setCursor(cursor_shape)
        for widget in self.fullscreen_overlay_watchers:
            widget.setCursor(cursor_shape)
        self.fullscreen_cursor_hidden = hidden

    def _draw_checkbox_icon(self, checked: bool, size: int = 18) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if checked:
            fill_color = QColor("#1d8cf8")
            border_color = QColor("#8fd1ff")
        else:
            fill_color = QColor(15, 23, 42, 140)
            border_color = QColor("#f8fafc")

        painter.setPen(QPen(border_color, 2))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(1, 1, size - 2, size - 2, 4, 4)

        if checked:
            painter.setPen(QPen(QColor("#ffffff"), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(4, size // 2, 7, size - 5)
            painter.drawLine(7, size - 5, size - 4, 4)

        painter.end()
        return QIcon(pixmap)

    def _draw_media_icon(self, kind: str, color_hex: str, size: int = 20) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(color_hex)
        pen = QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(color)

        if kind == "play":
            painter.drawPolygon(
                QPolygon(
                    [
                        QPoint(6, 4),
                        QPoint(6, 16),
                        QPoint(15, 10),
                    ]
                )
            )
        elif kind == "pause":
            painter.drawRoundedRect(5, 4, 3, 12, 1, 1)
            painter.drawRoundedRect(12, 4, 3, 12, 1, 1)
        elif kind == "fullscreen_enter":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(4, 8, 4, 4)
            painter.drawLine(4, 4, 8, 4)
            painter.drawLine(12, 4, 16, 4)
            painter.drawLine(16, 4, 16, 8)
            painter.drawLine(4, 12, 4, 16)
            painter.drawLine(4, 16, 8, 16)
            painter.drawLine(12, 16, 16, 16)
            painter.drawLine(16, 12, 16, 16)
        elif kind == "fullscreen_exit":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(8, 4, 8, 8)
            painter.drawLine(8, 8, 4, 8)
            painter.drawLine(12, 4, 12, 8)
            painter.drawLine(12, 8, 16, 8)
            painter.drawLine(4, 12, 8, 12)
            painter.drawLine(8, 12, 8, 16)
            painter.drawLine(12, 12, 16, 12)
            painter.drawLine(12, 12, 12, 16)

        painter.end()
        return QIcon(pixmap)

    def _update_playback_button_icon(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            main_icon = self._draw_media_icon("pause", "#102a43")
            overlay_icon = self._draw_media_icon("pause", "#f8fafc")
            self.play_pause_button.setToolTip("Pause")
            self.fullscreen_play_pause_button.setToolTip("Pause")
        else:
            main_icon = self._draw_media_icon("play", "#102a43")
            overlay_icon = self._draw_media_icon("play", "#f8fafc")
            self.play_pause_button.setToolTip("Play")
            self.fullscreen_play_pause_button.setToolTip("Play")
        self.play_pause_button.setIcon(main_icon)
        self.fullscreen_play_pause_button.setIcon(overlay_icon)

    def _update_fullscreen_button_icon(self, is_fullscreen: bool) -> None:
        if is_fullscreen:
            main_icon = self._draw_media_icon("fullscreen_exit", "#102a43")
            self.fullscreen_button.setToolTip("Exit Fullscreen")
        else:
            main_icon = self._draw_media_icon("fullscreen_enter", "#102a43")
            self.fullscreen_button.setToolTip("Enter Fullscreen")
        self.fullscreen_button.setIcon(main_icon)
        self.fullscreen_exit_button.setIcon(self._draw_media_icon("fullscreen_exit", "#f8fafc"))
        self.fullscreen_exit_button.setToolTip("Exit Fullscreen")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CSCI 576 Segment Player")
    app.setOrganizationName("CSCI576")
    window = PlayerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
