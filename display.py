import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QScrollArea, QPushButton, QHBoxLayout, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPainterPath, QFont, QFontMetrics, QColor
from PyQt5.QtCore import QTimer
import setproctitle

class OverlayWindow(QMainWindow):
    max_window_height = 500
    max_window_width = 1000

    def __init__(self):
        super().__init__()
        setproctitle.setproctitle("voiceAssistant")
        self.setWindowTitle("Voice Assistant")
        self.is_fullscreen = False
        self.normal_geometry = None
        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Initial window size
        self.width = 800
        self.height = 100
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(
            (screen.width() - self.width) // 2,
            0,  # Top of the screen
            self.width,
            self.height
        )

        # Create central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(5, 0, 5, 5)  # Remove top margin
        self.main_layout.setSpacing(2)  # Reduce spacing between buttons and scroll area

        # Create buttons
        self.exit_button = QPushButton("×", self.central_widget)
        self.minimize_button = QPushButton("−", self.central_widget)
        self.fullscreen_button = QPushButton("□", self.central_widget)

        # Style buttons
        button_style = """
            QPushButton {
                color: white;
                background-color: rgba(0, 0, 0, 150);
                border: none;
                border-radius: 2px;
                padding: 2px;
                min-width: 10px;
                min-height: 10px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 100);
            }
        """
        for button in [self.exit_button, self.minimize_button, self.fullscreen_button]:
            button.setStyleSheet(button_style)

        # Connect button signals
        self.exit_button.clicked.connect(self.close)
        self.minimize_button.clicked.connect(self.showMinimized)
        self.fullscreen_button.clicked.connect(self.toggleFullscreen)

        # Add buttons to a horizontal layout for top-left placement
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.exit_button)
        button_layout.addWidget(self.minimize_button)
        button_layout.addWidget(self.fullscreen_button)
        button_layout.addStretch()  # Push buttons to the left
        button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around buttons
        button_layout.setSpacing(5)  # Space between buttons

        # Create scroll area
        self.scroll_area = QScrollArea(self.central_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 50);
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 150);
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Create label for text display
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; background: transparent; padding: 5px 20px 5px 20px;")  # No top padding
        self.scroll_area.setWidget(self.label)

        # Add button layout and scroll area to main layout
        self.main_layout.addLayout(button_layout)
        self.main_layout.addWidget(self.scroll_area)

        # Initial text
        self.text = "Voice Assistant"
        self.updateText(self.text)

    def updateText(self, text):
        self.text = text

        # Adjust font size based on text length
        font_size = max(15, min(20, 75 // max(1, len(text) // 5)))
        font = QFont("Arial", font_size)
        self.label.setFont(font)
        self.label.setText(text)

        # Calculate text size
        scroll_padding = 20  # Account for left + right padding (40px each)
        text_max_width = self.max_window_width - scroll_padding

        # Constrain label's width and let it calculate height
        self.label.setFixedWidth(text_max_width)
        self.label.adjustSize()  # Forces QLabel to recalculate layout

        # Use sizeHint to calculate final window size
        content_height = self.label.sizeHint().height()
        self.width = self.max_window_width
        self.height = min(self.max_window_height, content_height + 30)  # Add space for buttons

        # Update geometry
        if not self.is_fullscreen:
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(
                (screen.width() - self.width) // 2,
                0,
                self.width,
                self.height
            )

        # Update scroll area geometry
        self.scroll_area.setFixedSize(self.width - 5, self.height - 30)  # Adjust for margins and buttons
        self.update()

    def toggleFullscreen(self):
        if self.is_fullscreen:
            self.setGeometry(self.normal_geometry)
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.is_fullscreen = False
            self.show()
        else:
            self.normal_geometry = self.geometry()
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.showFullScreen()
            self.is_fullscreen = True
        self.updateText(self.text)  # Update layout for fullscreen/normal mode

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create rounded rectangle path
        path = QPainterPath()
        radius = 10  # Corner radius
        path.addRoundedRect(QRectF(0, 0, self.width, self.height), radius, radius)

        # Draw translucent background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 125))  # Black with 50% opacity
        painter.drawPath(path)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()  # Close the dialog when Esc is pressed
        else:
            super().keyPressEvent(event)  # Pass other key events to the base class

def main():
    app = QApplication(sys.argv)
    overlay = OverlayWindow()

    # Example: Change text to demonstrate dynamic sizing
    overlay.updateText("Short Text")
    # QTimer.singleShot(2000, lambda: overlay.updateText("Medium sentence size. This is another sentence."))
    # QTimer.singleShot(4000, lambda: overlay.updateText(
    #     "THIS IS BEGIN. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.THIS IS END"))

    overlay.show()
    sys.exit(app.exec_())

def overlay_display(queue=None):
    app = QApplication(sys.argv)
    overlay = OverlayWindow()

    def check_queue():
        if queue and not queue.empty():
            new_text = queue.get()
            overlay.updateText(new_text)

    # Check queue every 500ms
    if queue:
        timer = QTimer()
        timer.timeout.connect(check_queue)
        timer.start(100)

    overlay.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
