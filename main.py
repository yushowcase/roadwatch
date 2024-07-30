import matplotlib

matplotlib.use('TkAgg')  # Set the backend to QtAgg before importing pyplot

from PyQt6.QtWidgets import QApplication
from ui import MainWindow

if __name__ == '__main__':
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()