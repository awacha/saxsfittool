import numpy as np
from PyQt5 import QtCore, QtGui


class FitParametersModel(QtCore.QAbstractItemModel):
    """A model storing fit parameters.

    Columns:
    (X) name | (X) lower bound | (X) upper bound | value | uncertainty | uncertainty percent

    (X) : has a tick/checker.

    The checker before the first column ("name") allows or disables
    fitting of the parameter. If fitting is allowed, "lower bound"
    and "upper bound" are checkable.

    """

    def __init__(self, parameters, unfittables):
        self._parameters = []
        for l in parameters:
            self._parameters.append({'name': l[0],
                                     'lowerbound': 0,
                                     'lowerbound_enabled': False,
                                     'upperbound_enabled': False,
                                     'lowerbound_active': False,
                                     'upperbound_active': False,
                                     'upperbound': 0,
                                     'value': 1,
                                     'uncertainty': 0,
                                     'description': l[1],
                                     'enabled': True,
                                     'fittable': True})
        for l in unfittables:
            self._parameters.append({'name':l[0],
                                     'description':l[1],
                                     'lowerbound': l[2],
                                     'upperbound': l[3],
                                     'lowerbound_enabled':True,
                                     'upperbound_enabled':True,
                                     'lowerbound_active':False,
                                     'upperbound_active':False,
                                     'value': l[4],
                                     'uncertainty':0,
                                     'enabled':False,
                                     'fittable':False,})
        super().__init__()

    def index(self, row, column, parent=None, *args, **kwargs):
        if column not in [0, 1, 2, 3, 4, 5]:
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
            return ['Name', 'Min.', 'Max.', 'Value', 'Uncertainty', 'Rel. unc. (%)'][column]

    def flags(self, modelindex):
        column = modelindex.column()
        row = modelindex.row()
        flagstoset = QtCore.Qt.ItemNeverHasChildren
        if 'fittable' not in self._parameters[row]:
            self._parameters[row]['fittable']=True
        if column == 0:
            # The name column is user-checkable.
            if self._parameters[row]['fittable']:
                flagstoset |= QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
            else:
                flagstoset |= QtCore.Qt.ItemIsEnabled
        elif column in [1, 2]:
            # lower and upper bound is user-checkable iff fitting is enabled
            if not self._parameters[row]['fittable']:
                flagstoset |= 0
            if self._parameters[row]['enabled']:
                flagstoset |= QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
            if (column == 1) and (self._parameters[row]['lowerbound_enabled']):
                flagstoset |= QtCore.Qt.ItemIsEditable
            elif (column == 2) and (self._parameters[row]['upperbound_enabled']):
                flagstoset |= QtCore.Qt.ItemIsEditable
        elif column == 3:
            # the value is always enabled and editable
            flagstoset |= QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
        elif column in [4,5]:
            if self._parameters[row]['enabled']:
                flagstoset |= QtCore.Qt.ItemIsEnabled

        return flagstoset

    def data(self, modelindex, role=None):
        row = modelindex.row()
        column = modelindex.column()
        if role is None:
            role = QtCore.Qt.DisplayRole
        if column == 0:
            if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
                return self._parameters[row]['name']
            elif role == QtCore.Qt.CheckStateRole:
                return [QtCore.Qt.Unchecked, QtCore.Qt.Checked][self._parameters[row]['enabled']]
        elif column == 1:
            if role == QtCore.Qt.DisplayRole:
                if self._parameters[row]['lowerbound_enabled']:
                    return str(self._parameters[row]['lowerbound'])
                else:
                    return 'Unlimited'
            elif role == QtCore.Qt.EditRole:
                assert self._parameters[row]['lowerbound_enabled']
                return self._parameters[row]['lowerbound']
            elif role == QtCore.Qt.CheckStateRole:
                return [QtCore.Qt.Unchecked, QtCore.Qt.Checked][self._parameters[row]['lowerbound_enabled']]
            elif role == QtCore.Qt.DecorationRole:
                return [None, QtGui.QIcon.fromTheme('dialog-warning')][self._parameters[row]['lowerbound_active']]
        elif column == 2:
            if role == QtCore.Qt.DisplayRole:
                if self._parameters[row]['upperbound_enabled']:
                    return str(self._parameters[row]['upperbound'])
                else:
                    return 'Unlimited'
            elif role == QtCore.Qt.EditRole:
                assert self._parameters[row]['upperbound_enabled']
                return self._parameters[row]['upperbound']
            elif role == QtCore.Qt.CheckStateRole:
                return [QtCore.Qt.Unchecked, QtCore.Qt.Checked][self._parameters[row]['upperbound_enabled']]
            elif role == QtCore.Qt.DecorationRole:
                return [None, QtGui.QIcon.fromTheme('dialog-warning')][self._parameters[row]['upperbound_active']]
        elif column == 3:
            if role == QtCore.Qt.DisplayRole:
                return str(self._parameters[row]['value'])
            elif role == QtCore.Qt.EditRole:
                return self._parameters[row]['value']
            elif role == QtCore.Qt.BackgroundRole:
                if ((self._parameters[row]['upperbound_enabled'] and
                             self._parameters[row]['value'] > self._parameters[row]['upperbound']) or
                        (self._parameters[row]['lowerbound_enabled'] and
                                 self._parameters[row]['value'] < self._parameters[row]['lowerbound'])):
                    return QtGui.QBrush(QtCore.Qt.red)
        elif column == 4:
            if role == QtCore.Qt.DisplayRole:
                if self._parameters[row]['enabled']:
                    return str(self._parameters[row]['uncertainty'])
                else:
                    return '(fixed)'
        elif column == 5:
            if role == QtCore.Qt.DisplayRole:
                if self._parameters[row]['enabled']:
                    if np.abs(self._parameters[row]['value'])<=np.finfo(self._parameters[row]['value']).eps:
                        return 'infinite'
                    else:
                        return '{:.2f} %'.format(np.abs(self._parameters[row]['uncertainty']/self._parameters[row]['value'])*100)
                else:
                    return '(fixed)'
        if role == QtCore.Qt.ToolTipRole:
            return self._parameters[row]['description']
        return None

    def setData(self, modelindex, data, role=QtCore.Qt.EditRole):
        row = modelindex.row()
        column = modelindex.column()
        if role is None:
            role = QtCore.Qt.EditRole
        if 'fittable' not in self._parameters[row]:
            self._parameters[row]['fittable']=True
        if role == QtCore.Qt.CheckStateRole:
            if column == 0 and self._parameters[row]['fittable']:
                self._parameters[row]['enabled'] = data == QtCore.Qt.Checked
                self.dataChanged.emit(modelindex, self.createIndex(row, self.columnCount() - 1, None))
            elif column == 1 and self._parameters[row]['fittable']:
                self._parameters[row]['lowerbound_enabled'] = data == QtCore.Qt.Checked
                self.dataChanged.emit(modelindex, self.createIndex(row, self.columnCount() - 1))
            elif column == 2 and self._parameters[row]['fittable']:
                self._parameters[row]['upperbound_enabled'] = data == QtCore.Qt.Checked
                self.dataChanged.emit(modelindex, self.createIndex(row, self.columnCount() - 1))
            else:
                return False
        elif role == QtCore.Qt.EditRole:
            if column == 1 and self._parameters[row]['fittable']:
                self._parameters[row]['lowerbound'] = float(data)
                self.dataChanged.emit(modelindex, self.createIndex(row, self.columnCount() - 1))
            elif column == 2 and self.parameters[row]['fittable']:
                self._parameters[row]['upperbound'] = float(data)
                self.dataChanged.emit(modelindex, self.createIndex(row, self.columnCount() - 1))
            elif column == 3:
                self._parameters[row]['value'] = float(data)
                self.dataChanged.emit(modelindex, self.createIndex(row, self.columnCount() - 1))
            else:
                return False
        else:
            return False
        return True

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, newparams):
        self.beginRemoveRows(QtCore.QModelIndex(), 0, len(self._parameters))
        self._parameters = []
        self.endRemoveRows()
        self.beginInsertRows(QtCore.QModelIndex(), 0, len(newparams))
        self._parameters = newparams
        for p in self._parameters:
            if 'fittable' not in p:
                p['fittable'] = True
        self.endInsertRows()

    def update_parameters(self, values, uncertainties):
        assert len(values) == len([p for p in self._parameters if p['fittable']])
        assert len(uncertainties) == len([p for p in self._parameters if p['fittable']])
        for i in range(len(values)):
            self._parameters[i]['value'] = values[i]
            self._parameters[i]['uncertainty'] = uncertainties[i]
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(), self.columnCount()))

    def emitParametersChanged(self):
        self.dataChanged.emit(self.createIndex(0,0), self.createIndex(self.rowCount(), self.columnCount()))

    def update_limits(self, lower=None, upper=None):
        fittablepars=[p for p in self._parameters if p['fittable']]
        if lower is None:
            lower = [None] * len( fittablepars)
        if upper is None:
            upper = [None] * len(fittablepars)
        for par, low, up in zip(fittablepars, lower, upper):
            if low is None or low == np.nan:
                par['lowerbound_enabled'] = False
                par['lowerbound'] = 0
            else:
                par['lowerbound_enabled'] = True
                par['lowerbound'] = low
            if up is None or up == np.nan:
                par['upperbound_enabled'] = False
                par['upperbound'] = 0
            else:
                par['upperbound_enabled'] = True
                par['upperbound'] = up
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(), self.columnCount()))

    def update_active_mask(self, active_mask):
        for par, am in zip(self._parameters, active_mask):
            par['upperbound_active'] = (am == 1)
            par['lowerbound_active'] = (am == -1)
        self.dataChanged.emit(self.createIndex(0, 1), self.createIndex(self.rowCount() - 1, 2),
                              [QtCore.Qt.DecorationRole])
