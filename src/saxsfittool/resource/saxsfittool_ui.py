# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/saxsfittool/resource/saxsfittool.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(756, 728)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.splitter_2 = QtWidgets.QSplitter(Form)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter = QtWidgets.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.inputFrame = QtWidgets.QFrame(self.splitter)
        self.inputFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.inputFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.inputFrame.setObjectName("inputFrame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.inputFrame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.inputdata_box = QtWidgets.QGroupBox(self.inputFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inputdata_box.sizePolicy().hasHeightForWidth())
        self.inputdata_box.setSizePolicy(sizePolicy)
        self.inputdata_box.setObjectName("inputdata_box")
        self.gridLayout = QtWidgets.QGridLayout(self.inputdata_box)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.inputdata_box)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 2)
        self.label = QtWidgets.QLabel(self.inputdata_box)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.maximumXDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.inputdata_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.maximumXDoubleSpinBox.sizePolicy().hasHeightForWidth())
        self.maximumXDoubleSpinBox.setSizePolicy(sizePolicy)
        self.maximumXDoubleSpinBox.setDecimals(4)
        self.maximumXDoubleSpinBox.setObjectName("maximumXDoubleSpinBox")
        self.gridLayout.addWidget(self.maximumXDoubleSpinBox, 2, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.inputdata_box)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.minimumXDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.inputdata_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.minimumXDoubleSpinBox.sizePolicy().hasHeightForWidth())
        self.minimumXDoubleSpinBox.setSizePolicy(sizePolicy)
        self.minimumXDoubleSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.minimumXDoubleSpinBox.setDecimals(4)
        self.minimumXDoubleSpinBox.setObjectName("minimumXDoubleSpinBox")
        self.gridLayout.addWidget(self.minimumXDoubleSpinBox, 1, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.inputdata_box)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.openButton = QtWidgets.QPushButton(self.inputdata_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openButton.sizePolicy().hasHeightForWidth())
        self.openButton.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon.fromTheme("document-open")
        self.openButton.setIcon(icon)
        self.openButton.setObjectName("openButton")
        self.gridLayout.addWidget(self.openButton, 0, 2, 1, 2)
        self.plotModeComboBox = QtWidgets.QComboBox(self.inputdata_box)
        self.plotModeComboBox.setObjectName("plotModeComboBox")
        self.gridLayout.addWidget(self.plotModeComboBox, 3, 2, 1, 2)
        self.setLimitsFromZoomPushButton = QtWidgets.QPushButton(self.inputdata_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.setLimitsFromZoomPushButton.sizePolicy().hasHeightForWidth())
        self.setLimitsFromZoomPushButton.setSizePolicy(sizePolicy)
        self.setLimitsFromZoomPushButton.setObjectName("setLimitsFromZoomPushButton")
        self.gridLayout.addWidget(self.setLimitsFromZoomPushButton, 1, 3, 2, 1)
        self.verticalLayout.addWidget(self.inputdata_box)
        self.fitcontrol_box = QtWidgets.QGroupBox(self.inputFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitcontrol_box.sizePolicy().hasHeightForWidth())
        self.fitcontrol_box.setSizePolicy(sizePolicy)
        self.fitcontrol_box.setObjectName("fitcontrol_box")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.fitcontrol_box)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.fitFunctionComboBox = QtWidgets.QComboBox(self.fitcontrol_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitFunctionComboBox.sizePolicy().hasHeightForWidth())
        self.fitFunctionComboBox.setSizePolicy(sizePolicy)
        self.fitFunctionComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.fitFunctionComboBox.setObjectName("fitFunctionComboBox")
        self.gridLayout_2.addWidget(self.fitFunctionComboBox, 0, 1, 2, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.executePushButton = QtWidgets.QPushButton(self.fitcontrol_box)
        self.executePushButton.setObjectName("executePushButton")
        self.horizontalLayout_2.addWidget(self.executePushButton)
        self.rePlotPushButton = QtWidgets.QPushButton(self.fitcontrol_box)
        self.rePlotPushButton.setObjectName("rePlotPushButton")
        self.horizontalLayout_2.addWidget(self.rePlotPushButton)
        self.plotModelPushButton = QtWidgets.QPushButton(self.fitcontrol_box)
        self.plotModelPushButton.setObjectName("plotModelPushButton")
        self.horizontalLayout_2.addWidget(self.plotModelPushButton)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 6, 0, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.fitcontrol_box)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 0, 0, 2, 1)
        self.algorithmComboBox = QtWidgets.QComboBox(self.fitcontrol_box)
        self.algorithmComboBox.setObjectName("algorithmComboBox")
        self.gridLayout_2.addWidget(self.algorithmComboBox, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.fitcontrol_box)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 2, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.fitcontrol_box)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 3, 0, 1, 1)
        self.lossFunctionComboBox = QtWidgets.QComboBox(self.fitcontrol_box)
        self.lossFunctionComboBox.setObjectName("lossFunctionComboBox")
        self.gridLayout_2.addWidget(self.lossFunctionComboBox, 3, 1, 1, 1)
        self.weightingCheckBox = QtWidgets.QCheckBox(self.fitcontrol_box)
        self.weightingCheckBox.setChecked(True)
        self.weightingCheckBox.setObjectName("weightingCheckBox")
        self.gridLayout_2.addWidget(self.weightingCheckBox, 5, 0, 1, 2)
        self.label_8 = QtWidgets.QLabel(self.fitcontrol_box)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 4, 0, 1, 1)
        self.yTransformComboBox = QtWidgets.QComboBox(self.fitcontrol_box)
        self.yTransformComboBox.setObjectName("yTransformComboBox")
        self.gridLayout_2.addWidget(self.yTransformComboBox, 4, 1, 1, 1)
        self.verticalLayout.addWidget(self.fitcontrol_box)
        self.fittingProgressBar = QtWidgets.QProgressBar(self.inputFrame)
        self.fittingProgressBar.setEnabled(True)
        self.fittingProgressBar.setMaximum(0)
        self.fittingProgressBar.setProperty("value", -1)
        self.fittingProgressBar.setTextVisible(False)
        self.fittingProgressBar.setInvertedAppearance(False)
        self.fittingProgressBar.setObjectName("fittingProgressBar")
        self.verticalLayout.addWidget(self.fittingProgressBar)
        self.parameters_box = QtWidgets.QGroupBox(self.inputFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.parameters_box.sizePolicy().hasHeightForWidth())
        self.parameters_box.setSizePolicy(sizePolicy)
        self.parameters_box.setObjectName("parameters_box")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.parameters_box)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.treeView = QtWidgets.QTreeView(self.parameters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeView.sizePolicy().hasHeightForWidth())
        self.treeView.setSizePolicy(sizePolicy)
        self.treeView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.treeView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.treeView.setAlternatingRowColors(True)
        self.treeView.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.treeView.setRootIsDecorated(False)
        self.treeView.setUniformRowHeights(True)
        self.treeView.setItemsExpandable(False)
        self.treeView.setObjectName("treeView")
        self.verticalLayout_2.addWidget(self.treeView)
        self.verticalLayout.addWidget(self.parameters_box)
        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.figure_widget = QtWidgets.QWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.figure_widget.sizePolicy().hasHeightForWidth())
        self.figure_widget.setSizePolicy(sizePolicy)
        self.figure_widget.setMinimumSize(QtCore.QSize(400, 0))
        self.figure_widget.setObjectName("figure_widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.figure_widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.figureLayout = QtWidgets.QVBoxLayout()
        self.figureLayout.setObjectName("figureLayout")
        self.verticalLayout_3.addLayout(self.figureLayout)
        self.verticalLayout_4.addWidget(self.figure_widget)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.reprFigureLayout = QtWidgets.QVBoxLayout()
        self.reprFigureLayout.setObjectName("reprFigureLayout")
        self.verticalLayout_6.addLayout(self.reprFigureLayout)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.correlationTableView = QtWidgets.QTableView(self.tab_3)
        self.correlationTableView.setObjectName("correlationTableView")
        self.verticalLayout_7.addWidget(self.correlationTableView)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.statisticsTreeView = QtWidgets.QTreeView(self.tab_4)
        self.statisticsTreeView.setObjectName("statisticsTreeView")
        self.verticalLayout_8.addWidget(self.statisticsTreeView)
        self.tabWidget.addTab(self.tab_4, "")
        self.logContainerWidget = QtWidgets.QWidget(self.splitter_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logContainerWidget.sizePolicy().hasHeightForWidth())
        self.logContainerWidget.setSizePolicy(sizePolicy)
        self.logContainerWidget.setMinimumSize(QtCore.QSize(0, 100))
        self.logContainerWidget.setObjectName("logContainerWidget")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.logContainerWidget)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.logContainerLayout = QtWidgets.QHBoxLayout()
        self.logContainerLayout.setObjectName("logContainerLayout")
        self.horizontalLayout_6.addLayout(self.logContainerLayout)
        self.verticalLayout_5.addWidget(self.splitter_2)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "SAXS Fit Tool"))
        self.inputdata_box.setTitle(_translate("Form", "Input data"))
        self.label_2.setText(_translate("Form", "Minimum x:"))
        self.label.setText(_translate("Form", "File:"))
        self.label_3.setText(_translate("Form", "Maximum x:"))
        self.label_5.setText(_translate("Form", "Plot mode:"))
        self.openButton.setText(_translate("Form", "Open"))
        self.setLimitsFromZoomPushButton.setText(_translate("Form", "Set from\n"
"zoom"))
        self.fitcontrol_box.setTitle(_translate("Form", "Fit control"))
        self.executePushButton.setText(_translate("Form", "Execute"))
        self.rePlotPushButton.setText(_translate("Form", "Replot"))
        self.plotModelPushButton.setText(_translate("Form", "Plot model"))
        self.label_4.setText(_translate("Form", "Model function:"))
        self.label_6.setText(_translate("Form", "Algorithm:"))
        self.label_7.setText(_translate("Form", "Loss function:"))
        self.weightingCheckBox.setText(_translate("Form", "Use y error bars for weights"))
        self.label_8.setText(_translate("Form", "Y transform:"))
        self.parameters_box.setTitle(_translate("Form", "Fit parameters"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Curves"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Model representation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "Parameter correlation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Form", "Results"))
