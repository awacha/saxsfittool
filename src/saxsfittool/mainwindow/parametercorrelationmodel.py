import matplotlib.cm
import numpy as np
from PyQt5 import QtCore, QtGui


class ParameterCorrelationModel(QtCore.QAbstractTableModel):
    def __init__(self, correlmatrix, parnames):
        super().__init__()
        self._correlmatrix = correlmatrix
        self._parnames = parnames

    def rowCount(self, parent=None, *args, **kwargs):
        return self._correlmatrix.shape[0]

    def columnCount(self, parent=None, *args, **kwargs):
        return self._correlmatrix.shape[1]

    def data(self, index: QtCore.QModelIndex, role=None):
        if role == QtCore.Qt.DisplayRole:
            return '{:.4f}'.format(self._correlmatrix[index.row(), index.column()])
        if role == QtCore.Qt.BackgroundRole:
            color = matplotlib.cm.RdYlGn_r(np.abs(self._correlmatrix[index.row(), index.column()]))
            return QtGui.QBrush(QtGui.QColor(*[int(component * 255) for component in color]))

    def headerData(self, idx, orientation, role=None):
        if role == QtCore.Qt.DisplayRole:
            return self._parnames[idx]
        return None
