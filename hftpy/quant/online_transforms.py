class ExponentialMovingAverage:
    def __init__(self,
                 input_feature_name,
                 alpha,
                 init_value,
                 required_n_warmups=100):
        self.input_name = input_feature_name
        self.alpha = alpha
        self.name = f'{input_feature_name}_ema_{self.alpha}'
        self.value = init_value
        self.required_n_warmups = required_n_warmups
        self.warmup_ema = None
        self.updates_since_init = 0

    def update(self, value):
        if self.value is not None:
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value
        else:
            if self.warmup_ema is None:
                self.warmup_ema = value
            else:
                self.warmup_ema = self.alpha * value + (1.0 - self.alpha) * self.warmup_ema
            if self.updates_since_init >= self.required_n_warmups:
                self.value = self.warmup_ema
            self.updates_since_init += 1
