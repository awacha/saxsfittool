import logging

import numpy as np
from PyQt5 import QtCore, QtWidgets

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent: QtWidgets.QWidget, options: QtWidgets.QStyleOptionViewItem,
                     index: QtCore.QModelIndex):
        parameter=index.model()._parameters[index.row()]
        if parameter['fittable']:
            if index.column() in [1, 2, 3]:
                logger.debug('Creating editor for fittable parameter {}'.format(parameter['name']))
                editor = QtWidgets.QDoubleSpinBox(parent)
                editor.setFrame(False)
                editor.setMinimum(-np.inf)
                editor.setMaximum(np.inf)
                editor.setDecimals(10)
                editor.setValue(parameter['value'])
            else:
                editor = super().createEditor(parent, options, index)
        else:
            if index.column() == 3:
                logger.debug('Creating editor for unfittable parameter {}'.format(parameter['name']))
                logger.debug('Bounds: {} to {}. Value: {}'.format(parameter['lowerbound'], parameter['upperbound'], parameter['value']))
                editor = QtWidgets.QSpinBox(parent)
                editor.setFrame(False)
                editor.setMinimum(parameter['lowerbound'])
                editor.setMaximum(parameter['upperbound'])
                editor.setValue(parameter['value'])
            else:
                editor = super().createEditor(parent, options, index)
        return editor

    def setEditorData(self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex):
        parameter=index.model()._parameters[index.row()]
        if parameter['fittable']:
            if index.column() in [1, 2, 3]:
                assert isinstance(editor, QtWidgets.QDoubleSpinBox)
                editor.setValue(float(index.model().data(index, QtCore.Qt.EditRole)))
            else:
                super().setEditorData(editor, index)
        else:
            if index.column() == 3:
                assert isinstance(editor, QtWidgets.QSpinBox)
                editor.setValue(index.model().data(index, QtCore.Qt.EditRole))
            else:
                super().setEditorData(editor, index)

    def setModelData(self, editor: QtWidgets.QWidget, model: QtCore.QAbstractItemModel, index: QtCore.QModelIndex):
        parameter=index.model()._parameters[index.row()]
        if parameter['fittable']:
            if index.column() in [1, 2, 3]:
                assert isinstance(editor, QtWidgets.QDoubleSpinBox)
                editor.interpretText()
                value = editor.value()
                model.setData(index, value, QtCore.Qt.EditRole)
            else:
                super().setModelData(editor, model, index)
        else:
            if index.column()==3:
                assert isinstance(editor, QtWidgets.QSpinBox)
                editor.interpretText()
                value = editor.value()
                model.setData(index, value, QtCore.Qt.EditRole)
            else:
                super().setModelData(editor, model, index)

    def updateEditorGeometry(self, editor: QtWidgets.QWidget, options: QtWidgets.QStyleOptionViewItem,
                             index: QtCore.QModelIndex):
        if index.column() in [1, 2, 3]:
            editor.setGeometry(options.rect)
        else:
            super().updateEditorGeometry(editor, options, index)
