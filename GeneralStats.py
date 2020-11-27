import numpy as np

class GeneralStats():
    def __init__(self, data):
        self.data = data
        self._min = None
        self._max = None
        self._range = None
        self._median = None
        self._mean = None
        self._std = None

    def get_statistics(self, axis):
        self.general_statistics(axis)
        print("Shape: ", self.data.shape)
        print("min: ", self._min, "\nmax: ", self._max, "\nrange: ", self._range)
        print("median: ", self._median, "\nmean: ", self._mean, "\nstd: ", self._std)


    def general_statistics(self, axis):
        self._shape = self.data.shape
        self.set_min(axis)
        self.set_max(axis)
        self.set_range(axis)
        self.set_median(axis)
        self.set_mean(axis)
        self.set_std(axis)

    def set_min(self, axis):
        self._min = np.amin(self.data, axis=axis)
    
    def set_max(self, axis):
        self._max = np.amax(self.data, axis=axis)

    def set_range(self, axis):
        self._range = np.ptp(self.data, axis=axis)

    def set_median(self, axis):
        self._median = np.median(self.data, axis=axis)

    def set_mean(self, axis):
        self._mean = np.mean(self.data, axis=axis)

    def set_std(self, axis):
        self._std = np.std(self.data, axis=axis)

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def get_range(self):
        return self._range
    
    def get_median(self):
        return self._median
    
    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std