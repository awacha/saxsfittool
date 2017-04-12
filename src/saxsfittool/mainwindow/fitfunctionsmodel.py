from PyQt5 import QtCore

from ..fitfunction import FitFunction


class FitFunctionsModel(QtCore.QAbstractItemModel):
    def __init__(self):
        super().__init__()
        self._functions = sorted(FitFunction.getsubclasses(), key=lambda x: x.name)

    def index(self, row, column, parent=None, *args, **kwargs):
        if column not in [0, 1]:
            raise ValueError('Invalid column: {}'.format(column))
        if row >= len(self._functions):
            raise ValueError('Invalid row: {}'.format(row))
        return self.createIndex(row, column, row * 2 + column)

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
            if modelindex.column() == 0:
                return self._functions[modelindex.row()].name
            elif modelindex.column() == 1:
                return self._functions[modelindex.row()]
        else:
            return None
