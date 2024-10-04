# online_transform.pyx

import numpy as np
np.import_array()

cdef class OnlineTransform:
    def __init__(self, str input_feature_name, double alpha, object init_value=None, bint adjust=True, int required_n_warmup=0):
        self.input_feature_name = input_feature_name
        self.alpha = alpha
        self.init_value = init_value
        self.adjust = adjust
        self.n = 0
        self.required_n_warmup = required_n_warmup
        self.value = None

    cpdef void update(self, double new_value):
        pass

    cpdef np.ndarray apply(self, np.ndarray data):
        cdef np.ndarray std_values = np.zeros(len(data), dtype=np.float64)
        cdef int i
        for i in range(len(data)):
            self.update(data[i])
            std_values[i] = self.value
        return std_values

    cpdef object warm_value(self):
        if self.n >= self.required_n_warmup:
            return self.value
        else:
            return None


cdef class ExponentialMA(OnlineTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('init_value') is not None:
            self.numerator = kwargs['init_value'] * 1.0
            self.denominator = 1.0
            self.value = kwargs['init_value']
        else:
            self.numerator = 0.0
            self.denominator = 0.0
            self.value = None

    cpdef void update(self, double value):
        cdef double numerator, denominator
        self.n += 1
        if not self.adjust:
            if self.value is None:
                self.value = value
            else:
                self.value = self.alpha * value + (1 - self.alpha) * self.value
        else:
            self.numerator = self.alpha * value + (1 - self.alpha) * self.numerator
            self.denominator = self.alpha + (1 - self.alpha) * self.denominator
            self.value = self.numerator / self.denominator


cdef class ExponentialSTD(OnlineTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.s = 0.0
        self.s2 = 0.0
        self.W = 0.0
        self.W2 = 0.0

    cpdef void update(self, double value):
        cdef double alpha, one_minus_alpha, m, v, denominator, adj
        self.n += 1
        alpha = self.alpha
        one_minus_alpha = 1 - alpha

        if self.n == 1:
            self.s = alpha * value
            self.s2 = alpha * value ** 2
            self.W = alpha
            self.W2 = alpha ** 2
        else:
            self.s = alpha * value + one_minus_alpha * self.s
            self.s2 = alpha * value ** 2 + one_minus_alpha * self.s2
            self.W = alpha + one_minus_alpha * self.W
            self.W2 = alpha ** 2 + one_minus_alpha ** 2 * self.W2

        m = self.s / self.W
        v = (self.s2 / self.W) - m ** 2

        if self.adjust:
            denominator = self.W ** 2 - self.W2
            if denominator != 0:
                adj = self.W ** 2 / denominator
            else:
                adj = 0
            v *= adj

        v = max(v, 0)
        self.value = np.sqrt(v) if self.W > 0 else np.nan
