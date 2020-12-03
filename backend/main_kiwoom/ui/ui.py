from kiwoom.kiwoom import *
import sys
from PyQt5.QtWidgets import *

class Ui_class():
    def __init__(self):
        print("ui class")


        self.app = QApplication(sys.argv)

        self.kiwoom = Kiwoom() ##인스턴트화 해야지 됨.

        self.app.exec_()


