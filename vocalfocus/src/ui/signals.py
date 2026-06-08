from PyQt6.QtCore import QObject, pyqtSignal


class AppSignals(QObject):
    
    #Central signal hub for the UI.

    # Live purification control
    start_live = pyqtSignal()
    stop_live = pyqtSignal()

    # When the user changes the focused speaker
    focus_changed = pyqtSignal(str)

    # When UI requests to open the Enroll New Speaker dialog
    request_enroll = pyqtSignal()

    # After a new profile has been enrolled
    profile_enrolled = pyqtSignal(str)

    # Simple log/status messages
    log_message = pyqtSignal(str)