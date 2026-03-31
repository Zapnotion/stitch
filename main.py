"""
main.py — Stitch entry point.

Run via:  python main.py
Or:       run.bat  (which sets up the venv first)
"""

import sys
import traceback
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox


def _install_exception_hook() -> None:
    """Route unhandled exceptions to the log file and show a dialog."""
    original = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        try:
            from app.backend.logger import log
            log.critical(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_tb),
            )
        except Exception:
            pass  # logger itself failed — fall through to original hook

        # Show a user-facing error dialog if a QApplication exists
        app = QApplication.instance()
        if app is not None:
            msg = QMessageBox()
            msg.setWindowTitle("Stitch — Unexpected Error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"An unexpected error occurred:\n\n{exc_value}")
            msg.setDetailedText("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            msg.exec()

        original(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


def main() -> int:
    _install_exception_hook()

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Stitch")
    app.setOrganizationName("Stitch")
    app.setApplicationVersion("0.1.0")

    # Log startup
    try:
        from app.backend.logger import log
        log.info("Stitch 0.1.0 starting")
    except Exception:
        pass

    from app.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
