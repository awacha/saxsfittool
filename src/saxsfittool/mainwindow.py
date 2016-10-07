from PyQt5 import QtWidgets, QtGui, uic, QtCore
from pkg_resources import resource_filename
import os
import numpy as np
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.axes import Axes

from .resource.saxsfittool_ui import Ui_Form

class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.dataset=None
        self.figure=None
        self.figureCanvas = None
        self.figureToolbar = None
        self.axes=None
        self.model=None
        self.plotModeModel = None
        self.setupUi(self)
        self.openButton.clicked.connect(self.openFile)
        self.show()

    def setupUi(self, Form):
        Ui_Form.setupUi(self, Form)
        Form.figure=Figure()
        Form.figureCanvas = FigureCanvasQTAgg(figure=Form.figure)
        Form.figureLayout.addWidget(Form.figureCanvas)
        Form.figureToolbar = NavigationToolbar2QT(Form.figureCanvas, Form)
        Form.figureLayout.addWidget(Form.figureToolbar)
        Form.axes=Form.figure.add_subplot(1,1,1)
        #ToDo: axes for residuals with subplot grid
        Form.model = QtGui.QStandardItemModel()
        Form.treeView.setModel(Form.model)
        Form.model.setHorizontalHeaderLabels(['Name', 'Lower bound', 'Upper bound', 'Value', 'Uncertainty'])
        Form.plotModeModel=QtGui.QStandardItemModel()
        Form.plotModeModel.appendColumn([QtGui.QStandardItem(x) for x in ['lin-lin','lin-log','log-lin','log-log']])
        Form.plotModeComboBox.setModel(Form.plotModeModel)
        Form.plotModeComboBox.setCurrentIndex(3)
        Form.minimumXDoubleSpinBox.valueChanged.connect(self.minimumXDoubleSpinBoxChanged)
        Form.maximumXDoubleSpinBox.valueChanged.connect(self.maximumXDoubleSpinBoxChanged)
        Form.plotModeComboBox.currentTextChanged.connect(self.rePlot)
        Form.rePlotPushButton.clicked.connect(self.rePlot)
        Form.setLimitsFromZoomPushButton.clicked.connect(self.setLimitsFromZoom)

    def setLimitsFromZoom(self):
        xmin,xmax=self.axes.get_xbound()
        self.minimumXDoubleSpinBox.setValue(xmin)
        self.maximumXDoubleSpinBox.setValue(xmax)

    def minimumXDoubleSpinBoxChanged(self):
        self.maximumXDoubleSpinBox.setMinimum(self.minimumXDoubleSpinBox.value())
        self.rePlot()

    def maximumXDoubleSpinBoxChanged(self):
        self.minimumXDoubleSpinBox.setMaximum(self.maximumXDoubleSpinBox.value())
        self.rePlot()

    def openFile(self):
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open curve...")
        if filename is None:
            return
        self.dataset=np.loadtxt(filename)
        # ToDo: handle exceptions in the previous statement
        self.minimumXDoubleSpinBox.setMinimum(self.dataX.min())
        self.minimumXDoubleSpinBox.setMaximum(self.dataX.max())
        self.minimumXDoubleSpinBox.setValue(self.dataX.min())
        self.maximumXDoubleSpinBox.setMinimum(self.dataX.min())
        self.maximumXDoubleSpinBox.setMaximum(self.dataX.max())
        self.maximumXDoubleSpinBox.setValue(self.dataX.max())
        stepsize = 10**(np.floor(np.log10((self.dataX.max()-self.dataX.min())/10)))
        self.maximumXDoubleSpinBox.setSingleStep(stepsize)
        self.minimumXDoubleSpinBox.setSingleStep(stepsize)
        self.rePlot()

    def rePlot(self):
        assert isinstance(self.axes, Axes)
        self.axes.clear()
        mask=self.dataMask
        self.axes.errorbar(self.roiX, self.roiY, self.roiDY, self.roiDX, 'b.')
        self.axes.plot(self.maskedX, self.maskedY, '.', color='gray')
        self.axes.grid(True, which='both')
        scaling = self.plotModeComboBox.currentText()
        if scaling.split('-')[0]=='lin':
            self.axes.set_xscale('linear')
        else:
            self.axes.set_xscale('log')
        if scaling.split('-')[1]=='lin':
            self.axes.set_yscale('linear')
        else:
            self.axes.set_yscale('log')
        self.figureCanvas.draw()

    @property
    def roiX(self):
        return self.dataset[:,0][self.dataMask]

    @property
    def roiY(self):
        return self.dataset[:,1][self.dataMask]

    @property
    def roiDY(self):
        try:
            return self.dataset[:,2][self.dataMask]
        except (IndexError, TypeError):
            return None

    @property
    def roiDX(self):
        try:
            return self.dataset[:,3][self.dataMask]
        except (IndexError, TypeError):
            return None

    @property
    def maskedX(self):
        return self.dataset[:,0][~self.dataMask]

    @property
    def maskedY(self):
        return self.dataset[:,1][~self.dataMask]

    @property
    def maskedDY(self):
        try:
            return self.dataset[:,2][~self.dataMask]
        except (IndexError, TypeError):
            return None

    @property
    def maskedDX(self):
        try:
            return self.dataset[:,3][~self.dataMask]
        except (IndexError, TypeError):
            return None


    @property
    def dataX(self):
        return self.dataset[:,0]

    @property
    def dataY(self):
        return self.dataset[:,1]

    @property
    def dataDY(self):
        try:
            return self.dataset[:,2]
        except (IndexError, TypeError):
            return None

    @property
    def dataDX(self):
        try:
            return self.dataset[:,3]
        except (IndexError, TypeError):
            return None

    @property
    def dataMask(self):
        return np.logical_and(self.dataX >= self.minimumXDoubleSpinBox.value(),
                              self.dataX <= self.maximumXDoubleSpinBox.value())


if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    app.exec_()
