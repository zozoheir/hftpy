from abc import abstractmethod

import numpy as np


class OnlineTransform:
    def __init__(self, input_feature_name, alpha, init_value=None, adjust=True, required_n_warmup=0):
        self.input_feature_name = input_feature_name
        self.alpha = alpha
        self.init_value = init_value
        self.adjust = adjust
        self.n = 0
        self.required_n_warmup = required_n_warmup
        self.value = None

    @abstractmethod
    def update(self, new_value):
        pass

    def apply(self, data):
        std_values = np.zeros(len(data))
        for i in range(len(data)):
            self.update(data[i])
            std_values[i] = self.value
        return std_values

    @property
    def warm_value(self):
        if self.n >= self.required_n_warmup:
            return self.value
        else:
            return None

class ExponentialMA(OnlineTransform):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if kwargs.get('init_value') is not None:
            self.numerator = kwargs['init_value'] * 1.0  # Assuming initial weight is 1.0
            self.denominator = 1.0
            self.value = kwargs['init_value']
        else:
            self.numerator = 0.0
            self.denominator = 0.0
            self.value = None

    def update(self, value):
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


class ExponentialSTD(OnlineTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.s = 0.0
        self.s2 = 0.0
        self.W = 0.0
        self.W2 = 0.0

    def update(self, value):
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
            self.W2 = alpha ** 2 + (one_minus_alpha ** 2) * self.W2

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



from collections import deque
import numpy as np

class MovingAverage(OnlineTransform):
    def __init__(self,
                 window_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.window = deque(maxlen=self.window_size)

    def update(self, value):
        self.window.append(value)
        self.value = np.mean(self.window)

    @property
    def warm_value(self):
        if len(self.window) == self.window_size:
            return self.value
        else:
            return None

class MovingStd(OnlineTransform):
    def __init__(self,
                 window_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.window = deque(maxlen=self.window_size)

    def update(self, value):
        self.window.append(value)
        if len(self.window) > 1:
            self.value = np.std(self.window, ddof=1)  # ddof=1 for sample standard deviation
        else:
            self.value = 0.0

    @property
    def warm_value(self):
        if len(self.window) == self.window_size:
            return self.value
        else:
            return None