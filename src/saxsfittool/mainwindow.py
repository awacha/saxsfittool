from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.axes import Axes

from .resource.saxsfittool_ui import Ui_Form
from .fitfunction import FitFunction

class SpinBoxDelegate(QtWidgets.QAbstractItemDelegate):
    def paint(self, painter, option, index):
        pass

    def sizeHint(self, option, index):
        pass


class FitFunctionsModel(QtCore.QAbstractItemModel):
    def __init__(self):
        super().__init__()
        self._functions=sorted(FitFunction.getsubclasses(), key=lambda x:x.name)
        print(self._functions)

    def index(self, row, column, parent=None, *args, **kwargs):
        if column not in [0,1]:
            raise ValueError('Invalid column: {}'.format(column))
        if row >= len(self._functions):
            raise ValueError('Invalid row: {}'.format(row))
        return self.createIndex(row, column, row*2+column)

    def parent(self, modelindex=None):
        return QtCore.QModelIndex()

    def rowCount(self, parent=None, *args, **kwargs):
        return len(self._functions)

    def columnCount(self, parent=None, *args, **kwargs):
        return 2

    def data(self, modelindex, role=None):
        if role is None:
            role = QtCore.Qt.DisplayRole
        if role == QtCore.Qt.DisplayRole:
            if modelindex.column()==0:
                return self._functions[modelindex.row()].name
            elif modelindex.column()==1:
                return self._functions[modelindex.row()]
        else:
            return None

class FitParametersModel(QtCore.QAbstractItemModel):
    def __init__(self, parameters):
        self._parameters=[]
        for l in parameters:
            self._parameters.append({'name':l[0],
                                     'lowerbound':-np.inf,
                                     'upperbound':np.inf,
                                     'value': 1,
                                     'uncertainty':0,
                                     'description': l[1],
                                     'enabled':True})
        super().__init__()

    def index(self, row, column, parent=None, *args, **kwargs):
        if column not in [0,1,2,3,4,5]:
            raise ValueError('Invalid column: {}'.format(column))
        if row >= len(self._parameters):
            raise ValueError('Invalid row: {}'.format(row))
        return self.createIndex(row, column, None)

    def parent(self, modelindex=None):
        return QtCore.QModelIndex()

    def rowCount(self, parent=None, *args, **kwargs):
        return len(self._parameters)

    def columnCount(self, parent=None, *args, **kwargs):
        return 6

    def headerData(self, column, orientation, role=None):
        if orientation == QtCore.Qt.Vertical:
            return None
        if role is None:
            role = QtCore.Qt.DisplayRole
        if role == QtCore.Qt.DisplayRole:
            return ['Fit?', 'Name', 'Min.', 'Max.', 'Value', 'Uncertainty'][column]

    def flags(self, modelindex):
        column = modelindex.column()
        row = modelindex.row()
        flags = QtCore.Qt.ItemNeverHasChildren
        if column == 0:
            flags |= QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
        if self._parameters[row]['enabled']:
            flags |=QtCore.Qt.ItemIsEnabled
            if column in [1,2,3]:
                flags |= QtCore.Qt.ItemIsEditable
        return flags

    def data(self, modelindex, role=None):
        row=modelindex.row()
        column=modelindex.column()
        if role is None:
            role = QtCore.Qt.DisplayRole
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            if column == 0:
                # parameter name
                return self._parameters[row]['name']
            elif column == 1:
                return str(self._parameters[row]['lowerbound'])
            elif column == 2:
                return str(self._parameters[row]['upperbound'])
            elif column == 3:
                return str(self._parameters[row]['value'])
            elif column == 4:
                return str(self._parameters[row]['uncertainty'])
            else:
                return None
        if role == QtCore.Qt.CheckStateRole:
            if column == 0:
                return [QtCore.Qt.Unchecked, QtCore.Qt.Checked][self._parameters[row]['enabled']]
        if role == QtCore.Qt.ToolTipRole:
            return self._parameters[row]['description']
        return None

    def setData(self, modelindex, data, role=QtCore.Qt.EditRole):
        row=modelindex.row()
        column=modelindex.column()
        if role is None:
            role = QtCore.Qt.EditRole
        print('setData: row: {}, col: {}, role: {}, data: {}, type(data): {}'.format(
            row, column, role, data, type(data)
        ))
        if role == QtCore.Qt.CheckStateRole:
            if column == 0:
                self._parameters[row]['enabled']=data==QtCore.Qt.Checked
                self.dataChanged.emit(modelindex, self.createIndex(row,self.columnCount()-1,None),
                                      [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.CheckStateRole])
            else:
                return False
        elif role == QtCore.Qt.EditRole:
            if column == 2:
                self._parameters[row]['lowerbound'] = float(data)
                self.dataChanged.emit(modelindex, modelindex, [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
            elif column == 3:
                self._parameters[row]['upperbound'] = float(data)
                self.dataChanged.emit(modelindex, modelindex, [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
            elif column == 4:
                self._parameters[row]['value'] = float(data)
                self.dataChanged.emit(modelindex, modelindex, [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
            else:
                return False
        else:
            return False
        return True


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
        Form.executePushButton.clicked.connect(self.doFitting)
        Form.fitfunctionsModel = FitFunctionsModel()
        Form.fitFunctionComboBox.setModel(Form.fitfunctionsModel)
        Form.fitFunctionComboBox.setCurrentIndex(0)
        Form.fitFunctionComboBox.currentIndexChanged.connect(self.fitFunctionSelected)
        Form.parameterModel = None
        self.fitFunctionSelected()

    def fitFunctionSelected(self):
        FitFunctionClass=self.fitfunctionsModel.data(
            self.fitfunctionsModel.index(self.fitFunctionComboBox.currentIndex(),1))
        print(FitFunctionClass)
        assert issubclass(FitFunctionClass, FitFunction)
        self.parameters_box.setTitle('Parameters for function "{}"'.format(FitFunctionClass.name))
        self.parametersModel = FitParametersModel(FitFunctionClass.arguments)
        self.treeView.setModel(self.parametersModel)

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

    def doFitting(self):
        pass


if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    app.exec_()
