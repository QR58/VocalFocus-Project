import sys
from pathlib import Path
from datetime import datetime


# Ensure the "src" directory is on sys.path so we can import core.*
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QMediaDevices
from PyQt6.QtGui import QPainter, QColor, QPen

from core.controllers import LiveController, EnrollmentController
from signals import AppSignals

from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QSizePolicy,
    QScrollArea,
    QTextEdit
)


# -----------------------------------
# Helper functions / widgets / Colors
# -----------------------------------

PRIMARY_PURPLE = "#6C3BCF"
SIDEBAR_PURPLE = "#47306E"
ACCENT_GREEN = "#10B981"
ACCENT_BLUE = "#29B6F6"
ACCENT_RED = "#E53935"
BACKGROUND = "#F3F3F3"
LIGHT_GRAY = "#DDDDDD"


def create_pill_label(text: str, bg: str = ACCENT_GREEN, fg: str = "white") -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedHeight(38)
    label.setStyleSheet(
        f"""
        QLabel {{
            background-color: {bg};
            color: {fg};
            border-radius: 19px;
            font-weight: bold;
            font-size: 16px;
        }}
        """
    )
    return label

def create_pill_label_widerSize(text: str, bg: str = ACCENT_GREEN, fg: str = "white") -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedHeight(40)
    label.setFixedWidth(300)
    label.setStyleSheet(
        f"""
        QLabel {{
            background-color: {bg};
            color: {fg};
            border-radius: 19px;
            font-weight: bold;
            font-size: 16px;
        }}
        """
    )
    return label


def create_pill_button(text: str, bg: str, fg: str = "white", width: int = 170, height: int = 40) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedSize(width, height)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {bg};
            color: {fg};
            border-color: black;
            border-width: 2px;
            border-style: solid;
            border-radius: 18px;
            padding: 4px 16px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: #8042ff;
            opacity: 0.9;
        }}
        """
    )
    return btn


def create_small_button(text: str, bg: str, fg: str = "white") -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(28)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {bg};
            color: black;
            border-color: black;
            border-width: 2px;
            border-style: solid;
            border-radius: 12px;
            padding: 2px 7px;
            font-size: 9px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {bg};
            opacity: 0.9;
        }}
        """
    )
    return btn

# -----------------------------
# UI widget for audio level display
# -----------------------------

class LevelMeter(QWidget):
    """
    Waveform-style level meter: many vertical bars inside a rounded black box.
    A controller feeds us a level in [0, 1]; we keep a short history
    and draw it as a stylized waveform.
    """
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._history_length = 60          # number of bars across the width
        self._levels = [0.0] * self._history_length

    def set_level(self, level: float) -> None:
        level = max(0.0, min(1.0, level))
        # push new value into history (scroll left)
        self._levels.pop(0)
        self._levels.append(level)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        radius = 18

        # --- background: same black rounded rectangle you had before ---
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(17, 17, 17))  # #111111
        painter.drawRoundedRect(rect, radius, radius)

        # --- draw waveform bars ---
        inner = rect.adjusted(16, 12, -16, -12)
        center_y = inner.center().y()
        max_half_height = inner.height() / 2.0

        if self._history_length <= 0:
            return

        bar_spacing = inner.width() / float(self._history_length)
        pen = QPen(QColor("#EEEEEE"))
        pen.setWidth(3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        for i, level in enumerate(self._levels):
            # slightly ease the levels so very small values still show a bit
            eased = level ** 0.7  # tweak shape; 0.7 keeps more mid–high
            half_h = max_half_height * eased

            x = inner.left() + (i + 0.5) * bar_spacing
            y1 = center_y - half_h
            y2 = center_y + half_h
            painter.drawLine(int(x), int(y1), int(x), int(y2))


# -----------------------------
# Live Console Page
# -----------------------------


class LiveConsolePage(QWidget):
    def __init__(self, signals: AppSignals, parent=None) -> None:
        super().__init__(parent)
        self.signals = signals
        self.is_running = False
        self.current_focus = "[None selected]"

        self._build_ui()
        self._connect_signals()

    # --- UI layout ---

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(18)

        # TOP: Input device & Focused speaker
        top_row = QHBoxLayout()
        top_row.setSpacing(40)
        main_layout.addLayout(top_row)

        # Input device
        input_layout = QVBoxLayout()
        self.input_header = create_pill_label("Input device")
        self.input_combo = QComboBox()
        # --- Populate with real audio input devices ---
        devices = QMediaDevices.audioInputs()
        if devices:
            default_device = QMediaDevices.defaultAudioInput()
            default_index = 0
            for idx, dev in enumerate(devices):
                self.input_combo.addItem(dev.description())
                if dev == default_device:
                    default_index = idx
            self.input_combo.setCurrentIndex(default_index)
        else:
            self.input_combo.addItem("No input devices found")
            
        self.input_combo.setFixedHeight(48)
        self.input_combo.setStyleSheet(
            """
            QComboBox {
                background-color: white;
                color: black;
                border-color: black;
                border-width: 2px;
                border-style: solid;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 18px;
            }
            
            QComboBox QAbstractItemView {
                 background-color: #6750a3;   /* list background color */
                 color: #EEEEEE;              /* item text color */
                 selection-background-color: #e2d4ff;
                 selection-color: white;
                }
            """
        )
        input_layout.addWidget(self.input_header)
        input_layout.addWidget(self.input_combo)
        top_row.addLayout(input_layout, 1)

        # Focused speaker
        focus_layout = QVBoxLayout()
        self.focus_header = create_pill_label("Focused speaker")
        self.focus_display = QLabel("[None selected]")
        self.focus_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.focus_display.setFixedHeight(48)
        self.focus_display.setStyleSheet(
            """
            QLabel {
                background-color: white;
                color: black;
                border-color: black;
                border-width: 2px;
                border-style: solid;
                border-radius: 16px;
                padding: 8px 18px;
                font-size: 18px;
                font-weight: bold;
            }
            """
        )
        focus_layout.addWidget(self.focus_header)
        focus_layout.addWidget(self.focus_display)
        top_row.addLayout(focus_layout, 1)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #052875;")
        main_layout.addWidget(line)

        # MIDDLE: Saved profiles + Input preview
        middle_row = QHBoxLayout()
        middle_row.setSpacing(40)
        main_layout.addLayout(middle_row, 3)

        # Saved profiles (left)
        profiles_layout = QVBoxLayout()
        self.saved_header = create_pill_label("Saved Profiles")
        profiles_layout.addWidget(self.saved_header)

        self.profile_table = QTableWidget(8, 4)
        self.profile_table.setHorizontalHeaderLabels(["Name", "Samples", "Created", "Last used"])
        self.profile_table.verticalHeader().setVisible(False)
        self.profile_table.horizontalHeader().setStretchLastSection(True)
        self.profile_table.setShowGrid(True)
        self.profile_table.setStyleSheet(
            """
            QTableWidget {
                background-color: white;
                border-radius: 8px;
                color: black;
            }
            QHeaderView::section {
                background-color: #E0E0E0;
                color: black;
                font-weight: 600;
            }
            """
        )
        header = self.profile_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        

        self.profile_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.profile_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        profiles_layout.addWidget(self.profile_table)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.enroll_btn = create_pill_button("Enroll New Speaker", bg="#B388FF")
        # self.delete_btn = create_pill_button("Delete", bg=ACCENT_RED)
        btn_row.addWidget(self.enroll_btn)
        # btn_row.addWidget(self.delete_btn)
        btn_row.addStretch()
        profiles_layout.addLayout(btn_row)
        middle_row.addLayout(profiles_layout, 3)

        # Input preview (right)
        preview_layout = QVBoxLayout()
        self.preview_header = create_pill_label("Input Preview")
        preview_layout.addWidget(self.preview_header)

        self.level_meter = LevelMeter()
        self.level_meter.setFixedHeight(130)
        preview_layout.addWidget(self.level_meter)

        self.preview_status = QLabel('Not running – click "Start Live Purification"')
        self.preview_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_status.setStyleSheet(
            """
            QLabel {
                color: Black;
                font-size: 14px;
            }
            """
        )
        preview_layout.addWidget(self.preview_status)
        middle_row.addLayout(preview_layout, 2)
        
        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #052875;")
        main_layout.addWidget(line)

        # BOTTOM: Output options, Start/Stop button, Enrollment tips + status bar
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(25)
        main_layout.addLayout(bottom_row, 2)

        # Output options
        output_layout = QVBoxLayout()
        self.output_header = create_pill_label("Output options")
        output_layout.addWidget(self.output_header)

        # Virtual Microphone radio (Future Work!).
        self.virtual_mic_radio = QRadioButton()
        self.virtual_mic_label = QLabel("Virtual Microphone")
        self.virtual_mic_label.setStyleSheet("color: black; font-size: 14px; font-weight: 600;")

        vm_row = QHBoxLayout()
        vm_row.addWidget(self.virtual_mic_radio)
        vm_row.addWidget(self.virtual_mic_label)
        vm_row.addStretch()
        output_layout.addLayout(vm_row)

        # --- Save to WAV radio ---
        self.save_wav_radio = QRadioButton()
        self.save_wav_label = QLabel("Save to WAV")
        self.save_wav_label.setStyleSheet("color: black; font-size: 14px; font-weight: 600;")

        wav_row = QHBoxLayout()
        wav_row.addWidget(self.save_wav_radio)
        wav_row.addWidget(self.save_wav_label)
        wav_row.addStretch()
        output_layout.addLayout(wav_row)
        
        # --- Radio Button Style ---
        self.virtual_mic_radio.setStyleSheet(
            """
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #6750a4;
                background-color: transparent;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #6750a4;
                background-color: #6750a4;
            }
        """
        )

        self.save_wav_radio.setStyleSheet(
            """
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #6750a4;
                background-color: transparent;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #6750a4;
                background-color: #6750a4;
            }
        """
        )

        output_layout.addStretch()
        bottom_row.addLayout(output_layout, 2)

        # Start/Stop button (center)
        center_layout = QVBoxLayout()
        center_layout.addStretch()
        self.start_stop_btn = QPushButton("Start Live Purification")
        self._set_start_button_style()
        self.start_stop_btn.setFixedSize(260, 60)
        center_layout.addWidget(self.start_stop_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        center_layout.addStretch()
        bottom_row.addLayout(center_layout, 2)

        # Enrollment tips + status bar (right)
        tips_layout = QVBoxLayout()
        self.tips_header = create_pill_label("Enrollment Tips & Status")
        tips_layout.addWidget(self.tips_header)

        tips_text = QLabel(
            "• Record in a quiet room.\n"
            "• Keep a constant mic distance.\n"
            "• Speak naturally for 5–10 seconds."
        )
        tips_text.setStyleSheet(
            """
            QLabel {
                color: Black;
                font-size: 14px;
                font-weight: bold;
            }
            """
        )
        tips_layout.addWidget(tips_text)

        self.status_bar_label = QLabel("Ready – microphone detected.")
        self.status_bar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_bar_label.setFixedHeight(32)
        self.status_bar_label.setStyleSheet(
            """
            QLabel {
                background-color: #D0D0D0;
                color: #8042ff;
                border-radius: 10px;
                padding: 4px 6px;
            }
            """
        )
        tips_layout.addWidget(self.status_bar_label)
        tips_layout.addStretch()
        bottom_row.addLayout(tips_layout, 3)

        self.setStyleSheet(f"background-color: {BACKGROUND};")

    # --- styling helpers ---

    def _set_start_button_style(self) -> None:
        self.start_stop_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: black;
                border-color: black;
                border-width: 2px;
                border-style: solid;
                border-radius: 18px;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #039BE5;
            }}
            """
        )

    def _set_stop_button_style(self) -> None:
        self.start_stop_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {ACCENT_RED};
                color: white;
                border-color: black;
                border-width: 2px;
                border-style: solid;
                border-radius: 18px;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #C62828;
            }}
            """
        )
        
    def update_profiles(self, rows: list[dict]) -> None:
        """
        Fill the Saved Profiles table with real profiles from profiles.json.

        rows: list of dicts with keys: name, samples, created, last_used
        """
        self.profile_table.clearContents()

        row_count = max(8, len(rows)) if rows else 8
        self.profile_table.setRowCount(row_count)

        for row, info in enumerate(rows):
            self.profile_table.setItem(row, 0, QTableWidgetItem(info.get("name", "")))
            self.profile_table.setItem(row, 1, QTableWidgetItem(info.get("samples", "")))
            self.profile_table.setItem(row, 2, QTableWidgetItem(info.get("created", "")))
            self.profile_table.setItem(row, 3, QTableWidgetItem(info.get("last_used", "")))

        # Re-apply highlight if we already have a focused speaker
        if self.current_focus and self.current_focus != "[None selected]":
            self.select_profile_by_name(self.current_focus)

    def select_profile_by_name(self, name: str) -> None:
        """
        Select a row by profile name and emit focus_changed.
        This ensures LiveController gets the correct target profile.
        """
        for r in range(self.profile_table.rowCount()):
            item = self.profile_table.item(r, 0)
            if item and item.text() == name:
                self.profile_table.selectRow(r)
                self._on_profile_selected()
                break


    # --- signals / behavior ---

    def _connect_signals(self) -> None:
        self.start_stop_btn.clicked.connect(self._on_start_stop_clicked)
        self.enroll_btn.clicked.connect(lambda: self.signals.request_enroll.emit())
        self.profile_table.itemSelectionChanged.connect(self._on_profile_selected)
        self.signals.focus_changed.connect(self._on_focus_changed)
        self.signals.start_live.connect(lambda: self.set_running(True))
        self.signals.stop_live.connect(lambda: self.set_running(False))

    def _on_start_stop_clicked(self) -> None:
        if self.is_running:
            self.signals.stop_live.emit()
        else:
            self.signals.start_live.emit()

    def set_running(self, running: bool) -> None:
        self.is_running = running
        if running:
            self._set_stop_button_style()
            self.start_stop_btn.setText("Stop Live Purification")
            self.preview_status.setText(f"Currently detected: {self.current_focus}")
            self.status_bar_label.setText("Live – audio is being purified.")
        else:
            self._set_start_button_style()
            self.start_stop_btn.setText("Start Live Purification")
            self.preview_status.setText('Not running – click "Start Live Purification"')
            self.status_bar_label.setText("Ready – microphone detected.")

    def _on_profile_selected(self) -> None:
        rows = self.profile_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        name_item = self.profile_table.item(row, 0)
        if not name_item:
            return
        name = name_item.text()
        self.signals.focus_changed.emit(name)

    def _on_focus_changed(self, name: str) -> None:
        self.current_focus = name
        self.focus_display.setText(name)
        # highlight corresponding row
        for r in range(self.profile_table.rowCount()):
            for c in range(self.profile_table.columnCount()):
                item = self.profile_table.item(r, c)
                if not item:
                    continue
                if item.column() == 0 and item.text() == name:
                    item.setBackground(Qt.GlobalColor.green)
                else:
                    item.setBackground(Qt.GlobalColor.white)
        if self.is_running:
            self.preview_status.setText(f"Currently detected: {name}")


# -----------------------------
# Profiles Page
# -----------------------------


class ProfilesPage(QWidget):
    def __init__(self, signals: AppSignals, parent=None) -> None:
        super().__init__(parent)
        self.signals = signals
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        header = create_pill_label_widerSize("Profiles & Enrollments")
        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.table = QTableWidget(8, 5)
        self.table.setHorizontalHeaderLabels(["Name", "Samples", "Created", "Last used", "Actions"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setShowGrid(True)
        self.table.setStyleSheet(
            """
            QTableWidget {
                background-color: white;
                color: black;
                border-radius: 10px;
            }
            QHeaderView::section {
                background-color: #E0E0E0;
                color: black;
                font-weight: 800;
            }
            """
        )
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        #This for making the Profiles table stretched.
        
        self.table.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding)
        self.table.verticalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.Stretch)


        layout.addWidget(self.table)

        bottom_buttons = QHBoxLayout()
        bottom_buttons.addStretch()
        self.enroll_btn = create_pill_button("Enroll New Speaker", "#B388FF")
        # Playback Test button (Future Work!).
        self.playback_btn = create_pill_button("Playback Test", ACCENT_GREEN)
        bottom_buttons.addWidget(self.enroll_btn)
        bottom_buttons.addWidget(self.playback_btn)
        bottom_buttons.addStretch()
        layout.addLayout(bottom_buttons)

        hint = QLabel("* All profiles stored locally on this device.")
        hint.setStyleSheet("color: #E53935;")
        layout.addWidget(hint)
        layout.addStretch()

        self.setStyleSheet(f"background-color: {BACKGROUND};")

        self.enroll_btn.clicked.connect(lambda: self.signals.request_enroll.emit())


    def _reenroll(self, row: int) -> None:
        self.signals.request_enroll.emit()

    def _delete(self, row: int) -> None:
        self.table.removeRow(row)

    def update_profiles(self, rows: list[dict]) -> None:
        """
        Fill the Profiles table with real profiles from profiles.json.

        rows: list of dicts with keys: name, samples, created, last_used
        """
        self.table.clearContents()
        self.table.setRowCount(0)

        row_count = max(8, len(rows)) if rows else 8
        self.table.setRowCount(row_count)

        for row, info in enumerate(rows):
            # Text columns
            self.table.setItem(row, 0, QTableWidgetItem(info.get("name", "")))
            self.table.setItem(row, 1, QTableWidgetItem(info.get("samples", "")))
            self.table.setItem(row, 2, QTableWidgetItem(info.get("created", "")))
            self.table.setItem(row, 3, QTableWidgetItem(info.get("last_used", "")))

            # Actions column with Re-enroll / Delete buttons
            actions_widget = QWidget()
            h = QHBoxLayout(actions_widget)
            h.setContentsMargins(25, 2, 4, 2)
            h.setSpacing(16)
            # Re-enroll button (Future Work!).
            reenroll_btn = create_small_button("Re-enroll", "#FFB300")
            delete_btn = create_small_button("Delete", ACCENT_RED)

            reenroll_btn.clicked.connect(lambda _, r=row: self._reenroll(r))
            delete_btn.clicked.connect(lambda _, r=row: self._delete(r))

            h.addWidget(reenroll_btn)
            h.addWidget(delete_btn)
            h.addStretch()

            self.table.setCellWidget(row, 4, actions_widget)

# -----------------------------
# Enroll New Speaker dialog
# -----------------------------


class EnrollDialog(QDialog):
    def __init__(self, signals: AppSignals, parent=None) -> None:
        super().__init__(parent)
        self.signals = signals
        self.setWindowTitle("Enroll New Speaker")
        self.setModal(True)
                
        self.setMinimumSize(900, 600)
        self.resize(1000, 650)

        self._build_ui()
        
        self.enroll_controller = EnrollmentController(
            level_meter=self.enroll_level_meter,
            device_combo=self.mic_combo,
            parent=self,
        )

    def _build_ui(self) -> None:
        #----------left grey card | right yellow tips card----------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ---------- GREEN TOP PILL ----------
        title_pill = create_pill_label_widerSize("Enroll New Speaker")
        title_pill.setFixedHeight(50)
        wrapper = QHBoxLayout()
        wrapper.addStretch()
        wrapper.addWidget(title_pill)
        wrapper.addStretch()
        main_layout.addLayout(wrapper)

        # ---------- ROW WITH 2 CARDS ----------
        cards_row = QHBoxLayout()
        cards_row.setSpacing(20)
        main_layout.addLayout(cards_row, 1)

        # ---------- LEFT: grey enrollment card ----------
        left_card = QFrame()
        left_card.setStyleSheet(
            """
            QFrame {
                background-color: #D9D9D9;      /* light grey card */
                border: 2px solid black;
                border-radius: 4px;
            }
            """
        )
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(18)

        # Speaker name
        name_label = QLabel("Speaker name")
        name_label.setStyleSheet("font-weight: bold; font-size: 18px; color: black; border: none;")
        left_layout.addWidget(name_label)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter your name")
        self.name_edit.setFixedHeight(32)
        self.name_edit.setStyleSheet(
            """
            QLineEdit {
                background-color: white;
                color: black;
                border-radius: 4px;
                padding: 4px 8px;
                border: 1px solid black;
            }
            """
        )
        left_layout.addWidget(self.name_edit)

        # Microphone
        left_layout.addSpacing(10)
        mic_label = QLabel("Microphone")
        mic_label.setStyleSheet("font-weight: bold; font-size: 18px; color: black; border: none;")
        left_layout.addWidget(mic_label)

        self.mic_combo = QComboBox()

        devices = QMediaDevices.audioInputs()
        if devices:
            default_device = QMediaDevices.defaultAudioInput()
            default_index = 0
            for idx, dev in enumerate(devices):
                self.mic_combo.addItem(dev.description())
                if dev == default_device:
                    default_index = idx
            self.mic_combo.setCurrentIndex(default_index)
        else:
            self.mic_combo.addItem("No input devices found")

        self.mic_combo.setFixedHeight(32)
        self.mic_combo.setStyleSheet(
            """
            QComboBox {
                background-color: white;
                color: black;
                border-radius: 4px;
                padding: 4px 8px;
                border: 1px solid black;
            }
            QComboBox QAbstractItemView {
                background-color: #6750a3;
                color: #EEEEEE;
                selection-background-color: #e2d4ff;
                selection-color: white;
                }
            """
        )
        left_layout.addWidget(self.mic_combo)

        # Enrollment recording
        left_layout.addSpacing(10)
        rec_label = QLabel("Enrollment recording")
        rec_label.setStyleSheet("font-weight: bold; font-size: 18px; color: black; border: none;")
        left_layout.addWidget(rec_label)

        # Record button centered
        rec_btn_row = QHBoxLayout()
        rec_btn_row.addStretch()
        self.record_btn = create_pill_button("Record 5–10 sec", "#B388FF", width=190, height=38)
        rec_btn_row.addWidget(self.record_btn)
        rec_btn_row.addStretch()
        left_layout.addLayout(rec_btn_row)

        # Timer centered
        self.timer_label = QLabel("00:00 / 10:00")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 13px; font-weight: bold; color: black; border: none;")
        left_layout.addWidget(self.timer_label)

        # Big waveform bar – use the same LevelMeter as Live console
        self.enroll_level_meter = LevelMeter()
        self.enroll_level_meter.setFixedHeight(100)
        left_layout.addWidget(self.enroll_level_meter)

        left_layout.addStretch()

        # Buttons at bottom (centered)
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        self.save_btn = create_pill_button("Save Profile", ACCENT_GREEN, width=140)
        self.cancel_btn = create_pill_button("Cancel", ACCENT_RED, width=140)
        buttons_layout.addWidget(self.save_btn)
        buttons_layout.addWidget(self.cancel_btn)
        buttons_layout.addStretch()
        left_layout.addLayout(buttons_layout)

        cards_row.addWidget(left_card, 3)

        # ---------- RIGHT: yellow Quick Tips card ----------
        tips_card = QFrame()
        tips_card.setStyleSheet(
            """
            QFrame {
                background-color: #FFF8DC;      /* light yellow */
                border: 2px solid black;
                border-radius: 4px;
            }
            """
        )
        tips_layout = QVBoxLayout(tips_card)
        tips_layout.setContentsMargins(18, 10, 18, 18)
        tips_layout.setSpacing(10)

        tips_header = QLabel("Quick Tips")
        tips_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tips_header.setStyleSheet(
            "font-weight: bold; font-size: 20px; color: #C62828; border: none;"
        )
        tips_layout.addWidget(tips_header)

        tips_text = QLabel(
            "• Avoid background noise where possible.\n\n"
            "• Speak continuously – no long pauses.\n\n"
            "• Use the same mic you will use during sessions."
        )
        tips_text.setWordWrap(True)
        tips_text.setStyleSheet("font-weight: 700; font-size: 14px; color: #0D47A1; border: none;")
        tips_layout.addWidget(tips_text)

        # Let the card size itself to the content height
        tips_card.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed
        )

        # Wrap the card in a column so it is vertically centered
        right_col = QVBoxLayout()
        right_col.addStretch()
        right_col.addWidget(tips_card)
        right_col.addStretch()

        cards_row.addLayout(right_col, 2)

        # Dialog background (light)
        self.setStyleSheet(f"QDialog {{ background-color: {BACKGROUND}; }}")

        # Simple behavior (no real recording; just UI)
        self.record_btn.clicked.connect(self._on_record_clicked)
        self.save_btn.clicked.connect(self._on_save)
        self.cancel_btn.clicked.connect(self._on_cancel)

    def _on_record_clicked(self) -> None:
        """
        Toggle the enrollment recording visualization on/off.
        """
        self.enroll_controller.toggle_recording()

        if self.enroll_controller.is_recording:
            self.record_btn.setText("Stop recording")
            self.timer_label.setText("00:05 / 10:00")
        else:
            self.record_btn.setText("Record 5–10 sec")
            self.timer_label.setText("00:00 / 10:00")

    def _on_save(self) -> None:
        from diarization import enrollment as enroll_db  # local import to avoid GUI startup cost

        # Stop recording (closes WAV file in AudioPipeline)
        self.enroll_controller.stop()

        requested_name = self.name_edit.text().strip() or "Unnamed"
        enroll_path = self.enroll_controller.last_recording_path

        saved_profile_name = None

        if (
            enroll_path is None
            or not enroll_path.exists()
            or enroll_path.stat().st_size == 0
        ):
            # No valid audio recorded
            self.signals.log_message.emit(
                "[Enroll] No valid enrollment audio was recorded; profile not saved."
            )
        else:
            try:
                profile = enroll_db.create_profile(
                    name=requested_name,
                    enroll_wav_path=str(enroll_path),
                )
                saved_profile_name = profile.name
                self.signals.log_message.emit(
                    f"[Enroll] Profile '{profile.name}' saved with audio '{enroll_path.name}'."
                )
            except Exception as exc:
                self.signals.log_message.emit(f"[Enroll] Failed to save profile: {exc}")

        # Only notify the rest of the UI if a profile was actually saved
        if saved_profile_name is not None:
            self.signals.profile_enrolled.emit(saved_profile_name)

        self.accept()
        
    def _on_cancel(self) -> None:
        self.enroll_controller.stop()
        self.reject()


# -----------------------------
# Tips & Logs Page
# -----------------------------


class TipsLogsPage(QWidget):
    def __init__(self, signals: AppSignals, parent=None) -> None:
        super().__init__(parent)
        self.signals = signals
        self._build_ui()

        # show log messages in the Logs area
        self.signals.log_message.connect(self.append_log)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(24)

        # --------- Tips section (top) ---------
        tips_header = create_pill_label_widerSize("Tips")
        layout.addWidget(tips_header, alignment=Qt.AlignmentFlag.AlignHCenter)

        tips_card = QFrame()
        tips_card.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid black;
            }
            """
        )
        tips_layout = QVBoxLayout(tips_card)
        tips_layout.setContentsMargins(16, 12, 16, 12)
        tips_layout.setSpacing(6)

        tips_info = QLabel(
            "• When Live, logs will show real-time events.\n"
            "• Use the Profiles page to manage enrolled speakers.\n"
            "• Output can be routed to a virtual microphone or saved as WAV.\n"
            "• Make sure the correct input device is selected before starting.\n"
            "• Use headphones to avoid echo and feedback while monitoring.\n"
            "• Keep a consistent mic distance for all enrolled speakers.\n"
            "• Re-enroll speakers if their voice changes significantly (e.g., illness).\n"
            "• Check the logs panel if audio seems incorrect or unexpectedly silent."
        )
        tips_info.setWordWrap(True)
        tips_info.setStyleSheet("font-size: 14px; font-weight: 600; color: #052875; border: none;")
        tips_layout.addWidget(tips_info)

        layout.addWidget(tips_card)

        # gap before Logs
        layout.addSpacing(16)

        # --------- Logs section (bottom) ---------
        logs_header = create_pill_label_widerSize("Logs")
        layout.addWidget(logs_header, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.logs_view = QTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setMinimumHeight(220)
        self.logs_view.setStyleSheet(
            """
            QTextEdit {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #CCCCCC;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
                color: black;
            }
            """
        )
        layout.addWidget(self.logs_view, 1)

        # page background
        self.setStyleSheet(f"background-color: #F3F3F3;")

    def append_log(self, msg: str) -> None:
        self.logs_view.append(msg)


# -----------------------------
# Main Window
# -----------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.signals = AppSignals()
        self.setWindowTitle("VocalFocus – AI-Powered Voice Purification")
        self.resize(1100, 700)

        self._build_ui()
        self._connect_signals()
        
        self.live_controller = LiveController(
            signals=self.signals,
            level_meter=self.live_page.level_meter,
            input_combo=self.live_page.input_combo,
            parent=self,
        )
        
        self._reload_profiles_from_db()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top bar (purple)
        top_bar = QFrame()
        top_bar.setFixedHeight(60)
        top_bar.setStyleSheet(
            f"""
            QFrame {{
                background-color: {PRIMARY_PURPLE};
                color: white;
            }}
            """
        )
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(16, 6, 16, 6)

        title_box = QVBoxLayout()
        title_lbl = QLabel("VocalFocus")
        title_lbl.setStyleSheet("font-weight: 800; font-size: 18px;")
        subtitle_lbl = QLabel("AI-Powered Voice Purification")
        subtitle_lbl.setStyleSheet("font-size: 11px;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        tb_layout.addLayout(title_box)

        tb_layout.addStretch()

        # Help button on top bar (Future Work).

        self.help_btn = QPushButton("Help")
        self.help_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                color: white;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                color: #10B981;
            }
            """
        )
        tb_layout.addWidget(self.help_btn)

        self.live_indicator = QLabel("● Idle")
        self.live_indicator.setStyleSheet("color: #00B0FF; font-weight: 700;")
        tb_layout.addWidget(self.live_indicator)

        root_layout.addWidget(top_bar)

        # Body: sidebar + stacked pages
        body_frame = QFrame()
        body_layout = QHBoxLayout(body_frame)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(170)
        sidebar.setStyleSheet(
            f"""
            QFrame#Sidebar {{
                background-color: {SIDEBAR_PURPLE};
                color: white;
            }}
            QPushButton.nav {{
                background: transparent;
                border: none;
                color: #D1C4E9;
                text-align: left;
                padding: 8px 22px;
                font-size: 18px;
            }}
            QPushButton.nav:checked {{
                color: {ACCENT_GREEN};
                font-weight: 700;
            }}
            """
        )
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 30, 0, 30)
        sb_layout.setSpacing(4)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.btn_live = QPushButton("Live Console")
        self.btn_profiles = QPushButton("Profiles")
        self.btn_tips_logs = QPushButton("Tips and Logs")
        for i, btn in enumerate((self.btn_live, self.btn_profiles, self.btn_tips_logs)):
            btn.setCheckable(True)
            btn.setProperty("class", "nav")
            btn.setObjectName(f"nav_{i}")
            btn.setStyleSheet("")
            self.nav_group.addButton(btn, i)
            sb_layout.addWidget(btn)

        sb_layout.addStretch()
        body_layout.addWidget(sidebar)

         # Pages
        self.stack = QStackedWidget()

        # Live Console wrapped in a scroll area so the bottom never gets cut off
        self.live_page = LiveConsolePage(self.signals)
        live_scroll = QScrollArea()
        live_scroll.setWidgetResizable(True)
        live_scroll.setFrameShape(QFrame.Shape.NoFrame)
        live_scroll.setWidget(self.live_page)

        self.profiles_page = ProfilesPage(self.signals)

        # Tips & Logs wrapped in a scroll area, like LiveConsolePage
        self.tips_logs_page = TipsLogsPage(self.signals)
        tips_logs_scroll = QScrollArea()
        tips_logs_scroll.setWidgetResizable(True)
        tips_logs_scroll.setFrameShape(QFrame.Shape.NoFrame)
        tips_logs_scroll.setWidget(self.tips_logs_page)
        
        # To make sure that the scroll area background is the same light grey
        tips_logs_scroll.setStyleSheet(
            "QScrollArea { background-color: #F3F3F3; border: none; }"
        )
        tips_logs_scroll.viewport().setStyleSheet("background-color: #F3F3F3;")

        self.stack.addWidget(live_scroll)         # index 0
        self.stack.addWidget(self.profiles_page)  # index 1
        self.stack.addWidget(tips_logs_scroll)    # index 2

        body_layout.addWidget(self.stack)
        root_layout.addWidget(body_frame)

        self.setCentralWidget(central)

        # Default page
        self.btn_live.setChecked(True)
        self.stack.setCurrentIndex(0)

        # Global background
        self.setStyleSheet(f"QMainWindow {{ background-color: {BACKGROUND}; }}")

    def _connect_signals(self) -> None:
        self.nav_group.idClicked.connect(self._on_nav_clicked)

        self.signals.start_live.connect(self._on_start_live)
        self.signals.stop_live.connect(self._on_stop_live)
        self.signals.log_message.connect(self._on_log_message)
        
        self.signals.profile_enrolled.connect(self._on_profile_enrolled)

    def _reload_profiles_from_db(self) -> None:
        """
        Load profiles from db/profiles.json and push them into both:
          - LiveConsolePage Saved Profiles table
          - ProfilesPage table
        """
        from diarization import enrollment as enroll_db  # local import to avoid circulars

        profiles = enroll_db.load_profiles()
        rows = []

        for p in profiles:
            p_path = Path(p.enroll_wav)
            samples = "1"
            created_str = "-"

            if p_path.exists():
                try:
                    ts = datetime.fromtimestamp(p_path.stat().st_mtime)
                    # Example: 08-Dec-2025 01:23
                    created_str = ts.strftime("%d-%b-%Y %H:%M")
                except Exception:
                    created_str = "-"

            rows.append(
                {
                    "name": p.name,
                    "samples": samples,
                    "created": created_str,
                    "last_used": "-",
                }
            )

        self.live_page.update_profiles(rows)
        self.profiles_page.update_profiles(rows)

        # If nothing is focused but we have at least one profile, focus the first one.
        if rows:
            current_focus = self.live_page.current_focus
            names = {row["name"] for row in rows}
            if current_focus == "[None selected]" or current_focus not in names:
                first_name = rows[0]["name"]
                self.live_page.select_profile_by_name(first_name)
                # This will emit focus_changed and update LiveController.


    def _on_profile_enrolled(self, name: str) -> None:
        """
        Called when EnrollDialog emits profile_enrolled(name).
        We reload the tables and auto-select this profile as the target
        speaker so purification will use it.
        """
        self._reload_profiles_from_db()
        self.live_page.select_profile_by_name(name)


    # Navigation
    def _on_nav_clicked(self, button_id: int) -> None:
        self.stack.setCurrentIndex(button_id)

    # Live state handling
    def _on_start_live(self) -> None:
        self.live_indicator.setText("● Live")
        self.live_indicator.setStyleSheet("color: #FF5252; font-weight: 700;")
        self.signals.log_message.emit("Live purification started")

    def _on_stop_live(self) -> None:
        self.live_indicator.setText("● Idle")
        self.live_indicator.setStyleSheet("color: #00B0FF; font-weight: 700;")
        self.signals.log_message.emit("Live purification stopped")

    def _on_log_message(self, msg: str) -> None:
        print("[LOG]", msg)

    # Public helper to open the Enroll dialog
    def show_enroll_dialog(self) -> None:
        dialog = EnrollDialog(self.signals, self)
        dialog.exec()

    # Override showEvent to connect enroll signal once window exists
    def showEvent(self, event) -> None:
        super().showEvent(event)
        # connect only once
        try:
            if not getattr(self, "_enroll_connected", False):
                self.signals.request_enroll.connect(self.show_enroll_dialog)
                self._enroll_connected = True
        except Exception:
            pass


# -----------------------------
# Entry point
# -----------------------------


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()