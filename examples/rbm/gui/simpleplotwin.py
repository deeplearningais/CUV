# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simpleplotwin.ui'
#
# Created: Wed Jun 16 16:33:15 2010
#      by: PyQt4 UI code generator 4.7.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_SimplePlotWindow(object):
    def setupUi(self, SimplePlotWindow):
        SimplePlotWindow.setObjectName("SimplePlotWindow")
        SimplePlotWindow.resize(483, 407)
        SimplePlotWindow.setFrameShape(QtGui.QFrame.StyledPanel)
        SimplePlotWindow.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout_2 = QtGui.QVBoxLayout(SimplePlotWindow)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.layout = QtGui.QVBoxLayout()
        self.layout.setObjectName("layout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.logscale = QtGui.QCheckBox(SimplePlotWindow)
        self.logscale.setObjectName("logscale")
        self.horizontalLayout.addWidget(self.logscale)
        self.updateButton = QtGui.QPushButton(SimplePlotWindow)
        self.updateButton.setObjectName("updateButton")
        self.horizontalLayout.addWidget(self.updateButton)
        self.wantAutomaticUpdate = QtGui.QPushButton(SimplePlotWindow)
        self.wantAutomaticUpdate.setCheckable(True)
        self.wantAutomaticUpdate.setChecked(True)
        self.wantAutomaticUpdate.setObjectName("wantAutomaticUpdate")
        self.horizontalLayout.addWidget(self.wantAutomaticUpdate)
        self.layout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.layout)

        self.retranslateUi(SimplePlotWindow)
        QtCore.QObject.connect(self.logscale, QtCore.SIGNAL("toggled(bool)"), self.updateButton.click)
        QtCore.QMetaObject.connectSlotsByName(SimplePlotWindow)

    def retranslateUi(self, SimplePlotWindow):
        SimplePlotWindow.setWindowTitle(QtGui.QApplication.translate("SimplePlotWindow", "Frame", None, QtGui.QApplication.UnicodeUTF8))
        self.logscale.setText(QtGui.QApplication.translate("SimplePlotWindow", "lo&g-scale", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setText(QtGui.QApplication.translate("SimplePlotWindow", "&update", None, QtGui.QApplication.UnicodeUTF8))
        self.wantAutomaticUpdate.setText(QtGui.QApplication.translate("SimplePlotWindow", "Automagic", None, QtGui.QApplication.UnicodeUTF8))

