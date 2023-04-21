import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensor_train
from tensor_train import TensorTrain

class Info:
    def __init__(self, label='None'):
        self._label = label.replace(' ', '\ ')
        self._errors = []
        self._negFrobenius = []
        self._negChebyshev = []
        self._negDensity = []
        self._minElements = {'true': [], 'found': []}

    def Reset(self):
        self._errors = []
        self._negFrobenius = []
        self._negChebyshev = []
        self._negDensity = []
        self._minElements = {'true': [], 'found': []}

    def ProcessTensorTrain(self, originalFullTensor, tt, foundMinimum=None):
        if isinstance(tt, tensor_train.TensorTrain):
            fullTensor = TensorTrain.GetFullTensor(tt.GetCores())
        else:
            fullTensor = TensorTrain.GetFullTensor(tt)

        self._AddError(np.linalg.norm(originalFullTensor - fullTensor) / np.linalg.norm(originalFullTensor))
        self._AddNegFrobenius(np.linalg.norm(fullTensor[fullTensor < 0]))
        self._AddNegChebyshev(np.abs(fullTensor.min()))
        self._AddNegDensity((fullTensor < 0).sum() / np.prod(fullTensor.shape))
        if foundMinimum is not None:
            self._minElements['true'].append(fullTensor.min())
            self._minElements['found'].append(foundMinimum)

    def GetErrors(self):
        return self._errors

    def GetNegFrobenius(self):
        return self._negFrobenius

    def GetNegChebyshev(self):
        return self._negFrobenius

    def GetNegDensity(self):
        return self._negDensity

    def GetMinElements(self):
        return self._minElements

    def PrintCurrentInfo(self):
        if self._errors:
            print('relative error:', self._errors[-1])
        if self._negFrobenius:
            print('negative elements (frobenius):', self._negFrobenius[-1])
        if self._negChebyshev:
            print('negative elements (chebyshev):', self._negChebyshev[-1])
        if self._negDensity:
            print('negative elements (density):', self._negDensity[-1])
        if self._minElements['true']:
            assert len(self._minElements['true']) == len(self._minElements['found'])
            print('minElement (true / found)', self._minElements['true'], '/', self._minElements['found'])

    def PlotErrors(self, figsize=(8, 4)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.grid()
        ax.set_title('$\\bf{%s}$\n%s' % (self._label, 'Relative Error'))
        ax.set_xlabel(r'iterations')

        ax.plot(range(1, len(self._errors) + 1), self._errors)

        plt.show()

    def PlotNegativeElements(self, figsize=(8, 4)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.grid()
        ax.set_yscale('log')
        ax.set_title('$\\bf{%s}$\n%s' % (self._label, 'Distance to nonnegative tensors'))
        ax.set_xlabel(r'iterations')

        ax.plot(range(1, len(self._negFrobenius) + 1), self._negFrobenius)

        plt.show()

    def _AddError(self, val):
        self._errors.append(val)

    def _AddNegFrobenius(self, val):
        self._negFrobenius.append(val)

    def _AddNegChebyshev(self, val):
        self._negChebyshev.append(val)

    def _AddNegDensity(self, val):
        self._negDensity.append(val)
