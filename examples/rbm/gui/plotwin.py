# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plotwin.ui'
#
# Created: Wed Jun 16 16:46:34 2010
#      by: PyQt4 UI code generator 4.7.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_PlotWindow(object):
    def setupUi(self, PlotWindow):
        PlotWindow.setObjectName("PlotWindow")
        PlotWindow.resize(360, 216)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PlotWindow.sizePolicy().hasHeightForWidth())
        PlotWindow.setSizePolicy(sizePolicy)
        PlotWindow.setMinimumSize(QtCore.QSize(120, 100))
        PlotWindow.setFrameShape(QtGui.QFrame.StyledPanel)
        PlotWindow.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout = QtGui.QVBoxLayout(PlotWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.layout = QtGui.QVBoxLayout()
        self.layout.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
        self.layout.setObjectName("layout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(PlotWindow)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.minibatchIdx = QtGui.QSpinBox(PlotWindow)
        self.minibatchIdx.setObjectName("minibatchIdx")
        self.horizontalLayout.addWidget(self.minibatchIdx)
        self.updateButton = QtGui.QPushButton(PlotWindow)
        self.updateButton.setObjectName("updateButton")
        self.horizontalLayout.addWidget(self.updateButton)
        self.wantAutomaticUpdate = QtGui.QPushButton(PlotWindow)
        self.wantAutomaticUpdate.setCheckable(True)
        self.wantAutomaticUpdate.setObjectName("wantAutomaticUpdate")
        self.horizontalLayout.addWidget(self.wantAutomaticUpdate)
        self.layout.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.layout)
        self.label.setBuddy(self.minibatchIdx)

        self.retranslateUi(PlotWindow)
        QtCore.QObject.connect(self.minibatchIdx, QtCore.SIGNAL("editingFinished()"), self.updateButton.click)
        QtCore.QMetaObject.connectSlotsByName(PlotWindow)

    def retranslateUi(self, PlotWindow):
        PlotWindow.setWindowTitle(QtGui.QApplication.translate("PlotWindow", "Plot", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("PlotWindow", "Idx", None, QtGui.QApplication.UnicodeUTF8))
        self.updateButton.setText(QtGui.QApplication.translate("PlotWindow", "&update", None, QtGui.QApplication.UnicodeUTF8))
        self.wantAutomaticUpdate.setText(QtGui.QApplication.translate("PlotWindow", "Automagic", None, QtGui.QApplication.UnicodeUTF8))

