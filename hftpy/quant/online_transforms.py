from abc import abstractmethod

import numpy as np
import pandas as pd


class OnlineTransform:
    def __init__(self, input_feature_name, alpha, init_value=None, adjust=True):
        self.input_feature_name = input_feature_name
        self.alpha = alpha
        self.init_value = init_value
        self.adjust = adjust
        self.n = 0

    @abstractmethod
    def update(self, new_value):
        pass

    def apply(self, data):
        std_values = []
        for value in data:
            self.update(value)
            std_values.append(self.value)
        return std_values



class ExponentialMA(OnlineTransform):
    def __init__(self, input_feature_name, alpha, init_value=None, adjust=True):
        super().__init__(input_feature_name, alpha, init_value, adjust)
        self.alpha = alpha
        self.adjust = adjust
        if init_value is not None:
            self.numerator = init_value * 1.0  # Assuming initial weight is 1.0
            self.denominator = 1.0
            self.value = init_value
        else:
            self.numerator = 0.0
            self.denominator = 0.0
            self.value = None

    def update(self, value):
        if not self.adjust:
            # Standard EMA without adjustment
            if self.value is None:
                self.value = value
            else:
                self.value = self.alpha * value + (1 - self.alpha) * self.value
        else:
            # Adjusted EMA
            self.numerator = self.alpha * value + (1 - self.alpha) * self.numerator
            self.denominator = self.alpha + (1 - self.alpha) * self.denominator
            self.value = self.numerator / self.denominator


class ExponentialSTD(OnlineTransform):
    def __init__(self, input_feature_name, alpha, init_value=None, adjust=True):
        super().__init__(input_feature_name, alpha, init_value, adjust)
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0
        self.W = 0.0
        self.W2 = 0.0
        self.value = None

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


class ExponentialZScore(OnlineTransform):
    def __init__(self, input_feature_name, alpha, init_value=None, adjust=True):
        super().__init__(input_feature_name, alpha, init_value, adjust)
        self.ema = ExponentialMA(input_feature_name, alpha, init_value, adjust)
        self.std = ExponentialSTD(input_feature_name, alpha, init_value, adjust)
        self.value = None

    def update(self, value):
        self.ema.update(value)
        self.std.update(value)
        if self.std.value is None or self.std.value == 0:
            self.value = 0  # Define z-score as 0 if std is zero or undefined to avoid division by zero
        else:
            self.value = (value - self.ema.value) / self.std.value


class ExponentialZScoreSTLT(OnlineTransform):
    def __init__(self, input_feature_name, st_alpha, lt_alpha, init_value=None, adjust=True):
        super().__init__(input_feature_name, None, init_value, adjust)
        self.ema_short = ExponentialMA(input_feature_name, st_alpha, init_value, adjust)
        self.ema_long = ExponentialMA(input_feature_name, lt_alpha, init_value, adjust)
        self.std_long = ExponentialSTD(input_feature_name, lt_alpha, init_value, adjust)
        self.value = None

    def update(self, value):
        self.ema_short.update(value)
        self.ema_long.update(value)
        self.std_long.update(value)
        if self.std_long.value is None or self.std_long.value == 0:
            self.value = 0  # Define z-score as 0 if std is zero or undefined to avoid division by zero
        else:
            self.value = (self.ema_short.value - self.ema_long.value) / self.std_long.value
