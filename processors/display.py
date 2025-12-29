import sys
import time
import setproctitle
import multiprocessing

from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QScrollArea, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout
)


class OverlayWindow(QMainWindow):
    max_window_width = 1000
    max_window_height = 500
    resize_border = 5  # Pixel width of resize border

    def __init__(self, text_queue=None):
        super().__init__()
        setproctitle.setproctitle("voiceAssistant")
        self.setWindowTitle("Voice Assistant")

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.dragging = False
        self.resizing = False
        self.drag_position = QPoint()
        self.resize_direction = None
        self.is_maximized = False
        self.text_queue = text_queue

        self.init_ui()

        if self.text_queue:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.check_text_queue)
            self.timer.start(100)

    def init_ui(self):
        self.default_width = 800
        self.default_height = 100
        self.resize(self.default_width, self.default_height)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - self.default_width) // 2, 0)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("""
            background: rgba(0, 0, 0, 150);
            border: 2px solid white;
            border-radius: 8px;
        """)

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(2, 0, 0, 5)
        button_layout.setAlignment(Qt.AlignLeft)

        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(20, 20)
        self.close_button.setStyleSheet(self.button_style())
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        self.minimize_button = QPushButton("−")
        self.minimize_button.setFixedSize(20, 20)
        self.minimize_button.setStyleSheet(self.button_style())
        self.minimize_button.clicked.connect(self.showMinimized)
        button_layout.addWidget(self.minimize_button)

        self.fullscreen_button = QPushButton("⛶")
        self.fullscreen_button.setFixedSize(20, 20)
        self.fullscreen_button.setStyleSheet(self.button_style())
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        button_layout.addWidget(self.fullscreen_button)

        main_layout.addLayout(button_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 0);
                width: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0);
                border: 1px solid white;
                border-radius: 5px;
            }
        """)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignLeft)
        self.label.setStyleSheet("""
            color: white;
            padding: 10px;
            background: rgba(0, 0, 0, 50);
            border: 1px solid white;
            border-radius: 5px;
            font-family: 'Courier New', 'Arial';
            font-size: 12px;
        """)
        self.scroll_area.setWidget(self.label)

        main_layout.addWidget(self.scroll_area)

        self.update_text("Voice Assistant")

    def button_style(self):
        return """
            QPushButton {
                background: rgba(0, 0, 0, 200);
                color: white;
                border: 1px solid white;
                border-radius: 8px;
                font-family: 'Courier New', 'Arial';
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 150);
            }
        """

    def update_text(self, text):
        self.label.setText(text)
        font_size = max(12, min(20, 75 // max(1, len(text) // 5)))
        font = QFont("Courier New", font_size, QFont.Normal)
        if not font.exactMatch():
            font = QFont("Courier New", font_size, QFont.Normal)
        self.label.setFont(font)

        if not self.is_maximized:
            # Calculate the actual height needed for the text
            font_metrics = QFontMetrics(font)
            # Account for padding in the label (10px on each side = 20px total)
            available_width = min(self.max_window_width, self.default_width) - 40  # 20px padding + margins
            
            # Calculate the bounding rectangle for the text with word wrapping
            text_rect = font_metrics.boundingRect(0, 0, available_width, 0, 
                                                Qt.AlignLeft | Qt.TextWordWrap, text)
            
            # Calculate required height: text height + label padding (20px) + button area + margins
            content_height = text_rect.height() + 20  # Label padding
            window_height = min(self.max_window_height, content_height + 60)  # +60 for buttons and margins
            window_width = min(self.max_window_width, self.default_width)
            
            # Set a minimum height to prevent the window from becoming too small
            window_height = max(window_height, 50)
            
            self.resize(window_width, window_height)

    def check_text_queue(self):
        if self.text_queue and not self.text_queue.empty():
            text = self.text_queue.get()
            if text == "":
                self.hide()
            else:
                self.show()
                self.update_text(text)

    def toggle_fullscreen(self):
        if self.is_maximized:
            self.showNormal()
            self.is_maximized = False
            self.fullscreen_button.setText("⛶")
            self.update_text(self.label.text())
        else:
            self.showMaximized()
            self.is_maximized = True
            self.fullscreen_button.setText("⤴")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            self.resize_direction = self.get_resize_direction(event.pos())
            if self.resize_direction:
                self.resizing = True
            else:
                self.dragging = True
            event.accept()

    def mouseMoveEvent(self, event):
        if self.resizing and event.buttons() == Qt.LeftButton:
            self.handle_resize(event.globalPos())
            event.accept()
        elif self.dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
        else:
            direction = self.get_resize_direction(event.pos())
            if direction == "left" or direction == "right":
                self.setCursor(Qt.SizeHorCursor)
            elif direction == "top" or direction == "bottom":
                self.setCursor(Qt.SizeVerCursor)
            elif direction in ["top-left", "bottom-right"]:
                self.setCursor(Qt.SizeFDiagCursor)
            elif direction in ["top-right", "bottom-left"]:
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.resizing = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()

    def get_resize_direction(self, pos):
        rect = self.rect()
        left = pos.x() < self.resize_border
        right = pos.x() > rect.width() - self.resize_border
        top = pos.y() < self.resize_border
        bottom = pos.y() > rect.height() - self.resize_border

        if top and left:
            return "top-left"
        elif top and right:
            return "top-right"
        elif bottom and left:
            return "bottom-left"
        elif bottom and right:
            return "bottom-right"
        elif left:
            return "left"
        elif right:
            return "right"
        elif top:
            return "top"
        elif bottom:
            return "bottom"
        return None

    def handle_resize(self, global_pos):
        if self.is_maximized:
            return

        rect = self.frameGeometry()
        new_rect = QRect(rect)

        if "left" in self.resize_direction:
            new_rect.setLeft(global_pos.x())
        if "right" in self.resize_direction:
            new_rect.setRight(global_pos.x())
        if "top" in self.resize_direction:
            new_rect.setTop(global_pos.y())
        if "bottom" in self.resize_direction:
            new_rect.setBottom(global_pos.y())

        new_width = min(max(new_rect.width(), 200), self.max_window_width)
        new_height = min(max(new_rect.height(), 100), self.max_window_height)
        new_rect.setWidth(new_width)
        new_rect.setHeight(new_height)

        self.setGeometry(new_rect)


def overlay_display(queue=None):
    app = QApplication(sys.argv)
    overlay = OverlayWindow(queue)

    def check_queue():
        if queue and not queue.empty():
            new_text = queue.get()
            if new_text == "":
                overlay.hide()
            else:
                overlay.show()
                overlay.update_text(new_text)

    if queue:
        timer = QTimer()
        timer.timeout.connect(check_queue)
        timer.start(100)

    overlay.hide()
    sys.exit(app.exec_())


def main():
    text_queue = multiprocessing.Queue()

    display_process = multiprocessing.Process(target=overlay_display, args=(text_queue,))
    display_process.start()

    # Simulate sending updates from main process
    for i in range(5):
        text_queue.put(f"Voice Assistant Update {i + 1}")
        time.sleep(2)

    display_process.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()