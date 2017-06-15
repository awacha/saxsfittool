import logging
import os
import pickle
import sys
import traceback
import textwrap
from concurrent.futures import ProcessPoolExecutor, Future

import numpy as np
import pkg_resources
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sastool.classes2.curve import Curve, FixedParameter

from .fitfunctionsmodel import FitFunctionsModel
from .fitparametersmodel import FitParametersModel
from .logviewer import LogViewer
from .parametercorrelationmodel import ParameterCorrelationModel
from .saxsfittool_ui import Ui_Form
from .spinboxdelegate import SpinBoxDelegate
from ..fitfunction import FitFunction
from .parameterstack import ParameterStack

logger = logging.getLogger(__name__)

class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.makeTestData()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.setSingleShot(False)
        self._fit_executor = ProcessPoolExecutor(max_workers=1)
        self._fit_future = None
        self._line_roi = None
        self._line_masked = None
        self._line_fitted = None
        self._line_residuals = None
        self._lastfitcurve = None
        self._lastfitstats = None
        self._parameterstack = ParameterStack()
        self._updating = False
        self.setupUi(self)
        logging.root.addHandler(self.logViewer)
        self.show()

    def curve(self):
        return Curve(self.roiX, self.roiY, self.roiDY, self.roiDX)

    def makeTestData(self):
        self.dataset = np.empty((100, 4), dtype=np.float)
        self.dataset[:, 0] = np.linspace(0.001, 100, self.dataset.shape[0])
        x0 = 30
        y0 = 10
        sigma = 5
        a = 100
        self.dataset[:, 1] = a / (2 * np.pi * sigma ** 2) ** 0.5 * np.exp(
            -(self.dataset[:, 0] - x0) ** 2 / (2 * sigma ** 2)) + y0
        self.dataset[:, 2] = self.dataset[:, 1] * 0.05
        self.dataset[:, 3] = (self.dataset[1, 0] - self.dataset[0, 0]) * 0.5
        self.dataset[:, 1] += np.random.randn(self.dataset.shape[0]) * self.dataset[:, 2]
        self.dataset[:, 0] += np.random.randn(self.dataset.shape[0]) * self.dataset[:, 3]

    def setupUi(self, Form):
        Ui_Form.setupUi(self, Form)
        Form.figure = Figure()
        Form.figureCanvas = FigureCanvasQTAgg(figure=Form.figure)
        Form.figureLayout.addWidget(Form.figureCanvas)
        Form.figureToolbar = NavigationToolbar2QT(Form.figureCanvas, Form)
        Form.figureLayout.addWidget(Form.figureToolbar)
        Form.initializeFigures()

        Form.figure_representation = Figure()
        Form.figureCanvas_representation = FigureCanvasQTAgg(figure=Form.figure_representation)
        Form.reprFigureLayout.addWidget(Form.figureCanvas_representation)
        Form.figureToolbar_representation = NavigationToolbar2QT(Form.figureCanvas_representation, Form)
        Form.reprFigureLayout.addWidget(Form.figureToolbar_representation)

        Form.plotModeModel = QtGui.QStandardItemModel()
        Form.plotModeModel.appendColumn([QtGui.QStandardItem(x) for x in ['lin-lin', 'lin-log', 'log-lin', 'log-log']])
        Form.plotModeComboBox.setModel(Form.plotModeModel)
        Form.plotModeComboBox.setCurrentIndex(3)
        Form.minimumXDoubleSpinBox.valueChanged.connect(Form.minimumXDoubleSpinBoxChanged)
        Form.maximumXDoubleSpinBox.valueChanged.connect(Form.maximumXDoubleSpinBoxChanged)
        Form.plotModeComboBox.currentTextChanged.connect(Form.rePlot)
        Form.rePlotPushButton.clicked.connect(lambda: Form.rePlot(full=True))
        Form.setLimitsFromZoomPushButton.clicked.connect(Form.setLimitsFromZoom)
        Form.executePushButton.clicked.connect(Form.doFitting)
        Form.fitfunctionsModel = FitFunctionsModel()
        Form.fitFunctionComboBox.setModel(Form.fitfunctionsModel)
        Form.fitFunctionComboBox.setCurrentIndex(0)
        Form.fitFunctionComboBox.currentIndexChanged.connect(Form.fitFunctionSelected)
        Form.parametersModel = None
        Form.algorithmComboBox.insertItems(0, ['trf', 'dogbox'])
        Form.lossFunctionComboBox.insertItems(0, ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan'])
        Form.logViewer = LogViewer(Form.logContainerWidget)
        Form.logContainerLayout.addWidget(Form.logViewer)
        Form._timer.timeout.connect(Form.checkIfFitIsDone)
        Form.plotModelPushButton.clicked.connect(Form.rePlotModel)
        Form.fittingProgressBar.hide()
        Form.fitFunctionSelected()
        Form.setLimits()
        Form.yTransformComboBox.insertItems(0, ['identity', 'ln'])
        Form.statisticsModel = QtGui.QStandardItemModel(0, 2)
        Form.statisticsModel.setHorizontalHeaderLabels(['Parameter', 'Value'])
        Form.statisticsTreeView.setModel(Form.statisticsModel)
        Form.exportResultsPushButton.clicked.connect(Form.exportResults)
        Form.rePlot()
        Form.loadParametersPushButton.clicked.connect(Form.loadParameters)
        Form.setWindowTitle(
            'SAXSFitTool v{} :: no file loaded yet'.format(pkg_resources.get_distribution('saxsfittool').version))
        self.openButton.clicked.connect(self.onFileSelected)
        Form.historySlider.valueChanged.connect(self.onHistorySlider)
        Form.clearHistoryPushButton.clicked.connect(self.onClearHistory)

    def onClearHistory(self):
        self._parameterstack.clear()
        self._parameterstack.push(self.parametersModel.parameters)
        self.updateHistorySlider()

    def onHistorySlider(self, value:int):
        if self._updating:
            return
        self._parameterstack.setPointer(value)
        self.setParameters(self._parameterstack.get())

    def setParameters(self, parameters):
        # print('setParameters')
        # for p in parameters:
        #     print(p['name'],p['value'])
        currentparams = self.parametersModel.parameters
        if not all([p['name'] == pc['name'] for p, pc in zip(parameters, currentparams)]):
            raise ValueError('The parameter file you selected is incompatible with the current model function. Parameters in the file: {}'.format(', '.join([p['name'] for p in parameters])))
        for pnew in parameters:
            pold = [p for p in self.parametersModel.parameters if p['name']==pnew['name']][0]
            for k in pnew:
                pold[k]=pnew[k]
        self.parametersModel.emitParametersChanged()

    def loadParameters(self):
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file to load parameters from...",
            '', filter='*.pickle',
        )
        if not filename:
            return
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            params = data['params']
            self.setParameters(params)
            self._parameterstack.push(params)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Error while loading file', exc.args[0])

    def exportResults(self):
        results = {}
        params = self.parametersModel.parameters
        results['params'] = params
        try:
            results['stats'] = self.laststats
        except AttributeError:
            pass
        results['dataset'] = {'x':self.roiX,
                              'y':self.roiY,
                              'dx':self.roiDX,
                              'dy':self.roiDY}
        filename, filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select file to save parameter & results pickle to...",
            os.path.splitext(self.windowFilePath())[0] + '.pickle', filter='*.pickle')
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        logger.info('Results saved to file {}'.format(filename))

    def fitFunctionSelected(self):
        FitFunctionClass = self.fitFunctionClass()
        assert issubclass(FitFunctionClass, FitFunction)
        self.parameters_box.setTitle('Parameters for function "{}"'.format(FitFunctionClass.name))
        self.parametersModel = FitParametersModel(FitFunctionClass.arguments, FitFunctionClass.unfittable_parameters)
        ff = FitFunctionClass()
        initpars = ff.initialize_arguments(self.roiX, self.roiY)
        if not isinstance(initpars, tuple):
            initpars = (initpars, None, None)
        self.parametersModel.update_parameters(initpars[0], [0] * len(initpars[0]))
        self.parametersModel.update_limits(initpars[1], initpars[2])
        self.treeView.setModel(self.parametersModel)
        self.treeView.setItemDelegate(SpinBoxDelegate())
        self._parameterstack.addModel(FitFunctionClass.name)
        self._parameterstack.setModel(FitFunctionClass.name)
        self._parameterstack.push(self.parametersModel.parameters)
        self.updateHistorySlider()

    def updateHistorySlider(self):
        self._updating=True
        try:
            self.historySlider.setMinimum(0)
            self.historySlider.setMaximum(len(self._parameterstack)-1)
            self.historySlider.setValue(self._parameterstack.pointer())
        finally:
            self._updating=False

    def fitFunctionClass(self):
        return self.fitfunctionsModel.data(
            self.fitfunctionsModel.index(self.fitFunctionComboBox.currentIndex(), 1)
        )

    def setLimitsFromZoom(self):
        xmin, xmax = self.axes.get_xbound()
        self.minimumXDoubleSpinBox.setValue(xmin)
        self.maximumXDoubleSpinBox.setValue(xmax)

    def minimumXDoubleSpinBoxChanged(self):
        self.maximumXDoubleSpinBox.setMinimum(self.minimumXDoubleSpinBox.value())
        self.rePlot()

    def maximumXDoubleSpinBoxChanged(self):
        self.minimumXDoubleSpinBox.setMaximum(self.maximumXDoubleSpinBox.value())
        self.rePlot()

    def onFileSelected(self):
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open curve...")
        if not filename:
            return
        self.openFile(filename)

    def openFile(self, filename):
        try:
            self.dataset = np.loadtxt(filename)
            self.dataset = self.dataset[np.isfinite(self.dataset.sum(axis=1)), :]
            self.fileNameLineEdit.setText(filename)
            self.setWindowFilePath(filename)
            self.setWindowTitle('SAXSFitTool v{} :: {}'.format(
                pkg_resources.get_distribution('saxsfittool').version,
                filename))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Error while opening file', str(exc),
                                           QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.NoButton)
            return
        self.setLimits()
        self.rePlot(full=True)

    def setLimits(self):
        self.minimumXDoubleSpinBox.setMinimum(self.dataX.min())
        self.minimumXDoubleSpinBox.setMaximum(self.dataX.max())
        self.minimumXDoubleSpinBox.setValue(self.dataX.min())
        self.maximumXDoubleSpinBox.setMinimum(self.dataX.min())
        self.maximumXDoubleSpinBox.setMaximum(self.dataX.max())
        self.maximumXDoubleSpinBox.setValue(self.dataX.max())
        stepsize = 10 ** (np.floor(np.log10((self.dataX.max() - self.dataX.min()) / 10)))
        self.maximumXDoubleSpinBox.setSingleStep(stepsize)
        self.minimumXDoubleSpinBox.setSingleStep(stepsize)
        self.rePlot()

    def rePlotModel(self):
        if self._line_fitted is not None:
            self._line_fitted.remove()
            self._line_fitted = None
        if self._line_residuals is not None:
            self._line_residuals.remove()
            self._line_residuals = None
        params = self.parametersModel.parameters
        val = [p['value'] for p in params]

        ff = self.fitFunctionClass()()
        assert isinstance(ff, FitFunction)
        fittedcurve = Curve(self.roiX, ff.function(self.roiX, *val))
        self._line_fitted, = fittedcurve.plot('r-', axes=self.axes, zorder=10)
        self._line_residuals, = (self.curve() - fittedcurve).plot('b.-', axes=self.axes_residuals)
        self.axes_residuals.set_xlim(*self.axes.get_xlim())
        self.axes_residuals.set_xscale(self.axes.get_xscale())
        self.axes_residuals.grid(True, which='both')
        self.figureCanvas.draw()
        ff.draw_representation(self.figure_representation, self.roiX,
                               *[p['value'] for p in self.parametersModel.parameters])
        self.figureCanvas_representation.draw()

    def initializeFigures(self):
        gs = GridSpec(4, 1, hspace=0)
        self.figure.clear()
        self.axes = self.figure.add_subplot(gs[:-1, :])
        self.axes.xaxis.set_ticks_position('top')
        self.axes.tick_params(labelbottom=False, labeltop=False)
        self.axes_residuals = self.figure.add_subplot(gs[-1, :], sharex=self.axes)
        self._line_residuals = None
        self._line_roi = None
        self._line_masked = None
        self._line_fitted = None

    def rePlot(self, full=False):
        assert isinstance(self.axes, Axes)
        if full:
            self.initializeFigures()
        if self._line_fitted is not None:
            self._line_fitted.remove()
            self._line_fitted = None
        if self._line_masked is not None:
            self._line_masked.remove()
            self._line_masked = None
        if self._line_roi is not None:
            self._line_roi.remove()
            self._line_roi = None
        mask = self.dataMask
        self._line_roi = self.axes.errorbar(self.roiX, self.roiY, self.roiDY, self.roiDX, 'b.', zorder=0)
        self._line_masked = self.axes.plot(self.maskedX, self.maskedY, '.', color='gray', zorder=0)[0]
        self.axes.autoscale(True, 'both', True)
        self.axes.grid(True, which='both')
        scaling = self.plotModeComboBox.currentText()
        if scaling.split('-')[0] == 'lin':
            self.axes.set_xscale('linear')
        else:
            self.axes.set_xscale('log')
        if scaling.split('-')[1] == 'lin':
            self.axes.set_yscale('linear')
        else:
            self.axes.set_yscale('log')
        self.axes_residuals.set_xlim(*self.axes.get_xlim())
        self.axes_residuals.set_xscale(self.axes.get_xscale())
        self.axes_residuals.grid(True, which='both')
        self.figureCanvas.draw()

    @property
    def roiX(self):
        return self.dataset[::self.decimationSpinBox.value()+1, 0][self.dataMask]

    @property
    def roiY(self):
        return self.dataset[::self.decimationSpinBox.value()+1, 1][self.dataMask]

    @property
    def roiDY(self):
        try:
            return self.dataset[::self.decimationSpinBox.value()+1, 2][self.dataMask]
        except (IndexError, TypeError):
            return None

    @property
    def roiDX(self):
        try:
            return self.dataset[::self.decimationSpinBox.value()+1, 3][self.dataMask]
        except (IndexError, TypeError):
            return None

    @property
    def maskedX(self):
        return self.dataset[::self.decimationSpinBox.value()+1, 0][~self.dataMask]

    @property
    def maskedY(self):
        return self.dataset[::self.decimationSpinBox.value()+1, 1][~self.dataMask]

    @property
    def maskedDY(self):
        try:
            return self.dataset[::self.decimationSpinBox.value()+1, 2][~self.dataMask]
        except (IndexError, TypeError):
            return None

    @property
    def maskedDX(self):
        try:
            return self.dataset[::self.decimationSpinBox.value()+1, 3][~self.dataMask]
        except (IndexError, TypeError):
            return None

    @property
    def dataX(self):
        return self.dataset[::self.decimationSpinBox.value()+1, 0]

    @property
    def dataY(self):
        return self.dataset[::self.decimationSpinBox.value()+1, 1]

    @property
    def dataDY(self):
        try:
            return self.dataset[::self.decimationSpinBox.value()+1, 2]
        except (IndexError, TypeError):
            return None

    @property
    def dataDX(self):
        try:
            return self.dataset[::self.decimationSpinBox.value()+1, 3]
        except (IndexError, TypeError):
            return None

    @property
    def dataMask(self):
        return np.logical_and(self.dataX >= self.minimumXDoubleSpinBox.value(),
                              self.dataX <= self.maximumXDoubleSpinBox.value())

    def workerfunc(self, fitfunc, *args, **kwargs):
        try:
            fitfunc(*args, **kwargs)
        except Exception as exc:
            print('Exception in workerfunc')
            print(traceback.format_exc())
            exc.tb=sys.exc_traceback()
            raise exc

    def doFitting(self):
        logger.info('Starting fit of dataset.')
        params = self.parametersModel.parameters
        self._parameterstack.push(params)
        self.updateHistorySlider()
        val = [[FixedParameter(p['value']), float(p['value'])][p['enabled']] for p in params if p['fittable']]
        unf = [float(p['value']) for p in params if not p['fittable']]
        if all([isinstance(x, FixedParameter) for x in val]):
            logger.error('Cannot fit with no free parameters.')
            return
        lbound = [[-np.inf, p['lowerbound']][p['lowerbound_enabled']] for p in params]
        ubound = [[np.inf, p['upperbound']][p['upperbound_enabled']] for p in params]
        self._fit_future = self._fit_executor.submit(self.curve().fit,
                                                     self.fitFunctionClass()().function,
                                                     val, unf, lbounds=lbound, ubounds=ubound,
                                                     ytransform=self.yTransformComboBox.currentText(),
                                                     loss=self.lossFunctionComboBox.currentText(),
                                                     method=self.algorithmComboBox.currentText())
        self.inputFrame.setEnabled(False)
        self.fittingProgressBar.show()
        self._timer.start()

    def checkIfFitIsDone(self):
        if self._fit_future is None or not self._fit_future.done():
            return False
        assert isinstance(self._fit_future, Future)
        try:
            exc = self._fit_future.exception()
            if exc is not None:
                logger.error('Error while fitting: {}'.format(exc))
                assert isinstance(exc, Exception)
                logger.error(exc.__traceback__)
                logger.error(exc.tb)
                return
            fitresults = self._fit_future.result()
            fitpars = fitresults[:-2]
            stats = fitresults[-2]
            if not stats['success']:
                logger.error('Fitting error: {}'.format(stats['message']))
                return
            func = self.fitFunctionClass()()
            pars = [p.val for p in fitpars]
            uncs = [p.err for p in fitpars]
            parsformatted = '\n'.join(
                ['{}: {:g} \xb1 {:g}'.format(name, val, unc)
                 for name, val, unc in zip([arg[0] for arg in func.arguments], pars, uncs)])
            correlmatrixformatted = '{}'.format(stats['Correlation_coeffs'])
            logger.info('Fitting completed successfully.\n'
                        '  Function: {0}\n'
                        '  X range: {1} to {2}\n'
                        '  Final parameter set:\n'
                        '{3}\n'
                        '  Correlation matrix:\n'
                        '{4}\n'
                        '  Message: {5[message]}\n'
                        '  Duration: {5[time]} seconds\n'
                        '  Number of function evaluations: {5[nfev]}\n'
                        '  Number of jacobian evaluations: {5[njev]}\n'
                        '  Optimality: {5[optimality]}\n'
                        '  Cost: {5[cost]}\n'
                        '  Status: {5[status]}\n'
                        '  Active_mask: {5[active_mask]}\n'
                        '  Original active_mask: {5[active_mask_original]}\n'
                        '  Chi2: {5[Chi2]}\n'
                        '  Reduced Chi2: {5[Chi2_reduced]}\n'
                        '  Degrees of freedom: {5[DoF]}\n'
                        '  R2: {5[R2]}\n'
                        '  Adjusted R2: {5[R2_adj]}\n'
                        '  R2 weighted by error bars: {5[R2_weighted]}\n'
                        '  Adjusted R2 weighted by error bars: {5[R2_adj_weighted]}\n'
                        '  CorMap test p-value: {5[CorMapTest_p]}\n'
                        '  CorMap test largest patch edge length: {5[CorMapTest_C]}\n'
                        '  CorMap test cormap size: {5[CorMapTest_n]}'
                        .format(self.fitFunctionClass().name,
                                self.roiX.min(), self.roiY.max(),
                                textwrap.indent(parsformatted, '    '),
                                textwrap.indent(correlmatrixformatted, '    '),
                                stats))
            self.statisticsModel.removeRows(0, self.statisticsModel.rowCount())
            for rowname, key in [('Duration (sec)', 'time'),
                                 ('Exit status', 'status'),
                                 ('Exit message', 'message'),
                                 ('Number of function evaluations', 'nfev'),
                                 ('Number of jacobian evaluations', 'njev'),
                                 ('Optimality', 'optimality'),
                                 ('Cost', 'cost'),
                                 ('Active_mask', 'active_mask'),
                                 ('Original active_mask', 'active_mask_original'),
                                 ('Degrees of freedom', 'DoF'),
                                 ('Chi2', 'Chi2'),
                                 ('Reduced Chi2', 'Chi2_reduced'),
                                 ('R2', 'R2'),
                                 ('Adjusted R2', 'R2_adj'),
                                 ('R2 weighted by errors in y', 'R2_weighted'),
                                 ('Adjusted R2 weighted by errors in y', 'R2_adj_weighted'),
                                 ('CorMap test P-value', 'CorMapTest_p'),
                                 ('CorMap test largest patch edge length', 'CorMapTest_C'),
                                 ('CorMap test cormap size', 'CorMapTest_n'),
                                 ]:
                self.statisticsModel.appendRow([QtGui.QStandardItem(rowname),
                                                QtGui.QStandardItem(str(stats[key]))])
            self.parametersModel.update_parameters(pars, uncs)
            self.correlationTableView.setModel(ParameterCorrelationModel(stats['Correlation_coeffs'],
                                                                         [arg[0] for arg in func.arguments]))
            self.parametersModel.update_active_mask(stats['active_mask'])
            self._parameterstack.push(self.parametersModel.parameters)
            self.updateHistorySlider()
            self.rePlotModel()
            self.laststats = stats
        finally:
            self._fit_future = None
            self.fittingProgressBar.hide()
            self.inputFrame.setEnabled(True)
            self._timer.stop()


