# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/saxsfittool/resource/logviewer.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(896, 283)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(0, 100))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.filterLevelComboBox = QtWidgets.QComboBox(Form)
        self.filterLevelComboBox.setObjectName("filterLevelComboBox")
        self.horizontalLayout.addWidget(self.filterLevelComboBox)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.keptMessagesSpinBox = QtWidgets.QSpinBox(Form)
        self.keptMessagesSpinBox.setKeyboardTracking(False)
        self.keptMessagesSpinBox.setMaximum(100000000)
        self.keptMessagesSpinBox.setProperty("value", 10000)
        self.keptMessagesSpinBox.setObjectName("keptMessagesSpinBox")
        self.horizontalLayout.addWidget(self.keptMessagesSpinBox)
        self.autoscrollCheckBox = QtWidgets.QCheckBox(Form)
        self.autoscrollCheckBox.setChecked(True)
        self.autoscrollCheckBox.setObjectName("autoscrollCheckBox")
        self.horizontalLayout.addWidget(self.autoscrollCheckBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.shownMessagesLabel = QtWidgets.QLabel(Form)
        self.shownMessagesLabel.setText("")
        self.shownMessagesLabel.setObjectName("shownMessagesLabel")
        self.horizontalLayout.addWidget(self.shownMessagesLabel)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 876, 231))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.logTreeView = QtWidgets.QTreeView(self.scrollAreaWidgetContents)
        self.logTreeView.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.logTreeView.setRootIsDecorated(False)
        self.logTreeView.setItemsExpandable(False)
        self.logTreeView.setObjectName("logTreeView")
        self.horizontalLayout_2.addWidget(self.logTreeView)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Filter level:"))
        self.label_2.setText(_translate("Form", "Number of entries to keep:"))
        self.autoscrollCheckBox.setText(_translate("Form", "Autoscroll on new message"))
        self.label_3.setText(_translate("Form", "Shown:"))

