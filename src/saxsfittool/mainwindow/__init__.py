import logging
import sys

from PyQt5 import QtWidgets

from .mainwindow import MainWindow


def run():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    try:
        mw.openFile(sys.argv[1])
    except:
        pass
    logging.root.setLevel(logging.DEBUG)
    app.exec_()


if __name__ == '__main__':
    run()
