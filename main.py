"""
main.py — Stitch entry point.

Run via:  python main.py
Or:       run.bat  (which sets up the venv first)
"""

import sys
import traceback
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox


def _suppress_native_stderr() -> None:
    """
    Qt's C++ layer and FFmpeg write directly to the Windows console handle,
    bypassing Python's sys.stderr entirely. The only reliable fix is to
    redirect file descriptor 2 (the real stderr fd) to NUL at the OS level,
    then re-open a Python-level stderr that writes to the log file instead.

    On non-Windows this is a no-op — the sys.stderr filter in the else branch
    is sufficient for Linux/macOS.
    """
    import os

    if sys.platform == "win32":
        import ctypes, ctypes.wintypes

        # Open NUL device
        nul_handle = ctypes.windll.kernel32.CreateFileW(
            "NUL",
            0x40000000,   # GENERIC_WRITE
            0x3,          # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            3,            # OPEN_EXISTING
            0, None,
        )

        # Replace the Windows STD_ERROR_HANDLE so Qt's C++ writes go to NUL
        STD_ERROR_HANDLE = ctypes.c_ulong(0xFFFFFFF4)
        ctypes.windll.kernel32.SetStdHandle(STD_ERROR_HANDLE, nul_handle)

        # Also redirect the low-level fd 2 so FFmpeg subprocess writes are swallowed
        try:
            nul_fd = os.open("NUL", os.O_WRONLY)
            os.dup2(nul_fd, 2)
            os.close(nul_fd)
        except OSError:
            pass

        # Keep a Python-level stderr pointing at the log file so our own
        # print() / logging still works
        try:
            from app.config import LOGS_DIR
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_path = LOGS_DIR / "stitch.log"
            # Open in append mode, line-buffered
            _log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
            sys.stderr = _log_fh
        except Exception:
            # If we can't open the log file just swallow stderr entirely
            sys.stderr = open(os.devnull, "w")

    else:
        # Non-Windows: filter noisy prefixes at the Python level
        _real_stderr = sys.stderr
        _SKIP = (
            "Input #", "  Duration:", "  Stream #", "    Metadata:",
            "    encoder", "[mp3float", "[mp3 @", "qt.multimedia",
            "Unknown property",
        )

        class _Filter:
            def write(self, text):
                if text.strip() and not any(text.lstrip().startswith(p) for p in _SKIP):
                    _real_stderr.write(text)
            def flush(self): _real_stderr.flush()
            def __getattr__(self, n): return getattr(_real_stderr, n)

        sys.stderr = _Filter()


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
            pass

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


def _suppress_qt_warnings() -> None:
    """Suppress Qt's own debug/warning output (e.g. 'Unknown property')."""
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    def _handler(msg_type, context, message):
        # Drop stylesheet warnings and multimedia noise; keep Critical/Fatal
        if msg_type in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            try:
                from app.backend.logger import log
                log.error(f"Qt: {message}")
            except Exception:
                pass

    qInstallMessageHandler(_handler)


def main() -> int:
    # Must happen before QApplication is created
    _suppress_native_stderr()
    _install_exception_hook()

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Stitch")
    app.setOrganizationName("Stitch")
    app.setApplicationVersion("0.1.0")

    # Install Qt message handler AFTER QApplication exists
    _suppress_qt_warnings()

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
