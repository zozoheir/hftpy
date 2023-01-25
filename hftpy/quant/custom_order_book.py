import numpy as np

#TODO Parallelize numpy array operations with jit or numba


class CustomOrderBook:

    def __init__(self,
                 n_levels=10,
                 exchange_timestamp_feature_name='exchange_timestamp',
                 receipt_timestamp_feature_name='receipt_timestamp',
                 optional_features=[]):

        self.n_levels = n_levels
        self.exchange_timestamp_column = exchange_timestamp_feature_name
        self.receipt_timestamp_column = receipt_timestamp_feature_name
        self.optional_features = optional_features

        # Historical data and live data have different names for order book amount columns
        self.bid_prices = np.zeros(self.n_levels, dtype=np.float)
        self.bid_quantity = np.zeros(self.n_levels, dtype=np.float)
        self.ask_prices = np.zeros(self.n_levels, dtype=np.float)
        self.ask_quantity = np.zeros(self.n_levels, dtype=np.float)
        self.mid = np.float('nan')
        self.tob_spread_absolute = np.float('nan')
        self.tob_spread_bps = np.float('nan')
        self.n_levels = n_levels
        self.total_bid_size = np.float('nan')
        self.total_ask_size = np.float('nan')
        self.bid_price = np.float('nan')
        self.ask_price = np.float('nan')
        self.receipt_timestamp = np.float('nan')
        self.exchange_timestamp = np.float('nan')

        if 'sequence_number' in optional_features:
            self.sequence_number_feature_name = 'sequence_number'
            self.last_sequence_number = -2e9
            self.sequence_number = 0

    def extractOB(self):
        """
        To implement in child classes
        :return:
        """

        return None

    def extractTimestampSequence(self):
        """
        To implement in child classes
        :return:
        """
        return None

    def update(self, ob_dict):
        self.ob_dict = ob_dict
        self.extractOB()
        self.extractTimestampSequence()
        self.computeMid()
        self.computeAbsoluteSpread()
        self.computeBpsSpread()

    def computeMid(self):
        if (not np.isnan(self.bid_price)) and (not np.isnan(self.ask_price)):
            self.mid = 0.5 * (self.bid_price + self.ask_price)
        else:
            self.mid = np.float('nan')

    def computeAbsoluteSpread(self):
        if (self.bid_prices[0] != 0.0) and (self.ask_prices[0] != 0.0):
            self.tob_spread_absolute = self.ask_prices[0] - self.bid_prices[0]
        else:
            self.tob_spread_absolute = np.float('nan')

    def computeBpsSpread(self):
        self.computeAbsoluteSpread()
        if np.isnan(self.mid):
            self.tob_spread_bps = np.float('nan')
        else:
            self.tob_spread_bps = self.tob_spread_absolute / self.mid * np.float(10000)

    def getVWSpreadAbs(self, volume):
        bid_vwap = self.c_get_bid_vwap(volume, False)
        ask_vwap = self.c_get_ask_vwap(volume, False)
        return ask_vwap - bid_vwap

    def getVWSpreadBps(self, volume):
        vw_spread = self.getVWSpreadAbs(volume)
        return vw_spread / self.mid * np.float(10000.0)

    def getBidVWAP(self, size, is_executable):
        cum_size = np.float(0.0)
        int_sizes = np.zeros(self.n_levels, dtype=np.float)

        for i in range(self.n_levels):
            if (self.bid_quantity[i] + cum_size) < size:
                int_sizes[i] = self.bid_quantity[i]
                cum_size += self.bid_quantity[i]
            else:
                int_sizes[i] = size - cum_size
                cum_size += size - cum_size

        if is_executable and (cum_size < size):
            return np.float('nan')
        else:
            return np.average(self.bid_prices, weights=int_sizes)

    def getAskVWAP(self, size, is_executable):
        cum_size = np.float(0.0)
        int_sizes = np.zeros(self.n_levels, dtype=np.float)
        for i in range(self.n_levels):
            if (self.ask_quantity[i] + cum_size) < size:
                int_sizes[i] = self.ask_quantity[i]
                cum_size += self.ask_quantity[i]
            else:
                int_sizes[i] = size - cum_size
                cum_size += size - cum_size

        if is_executable and (cum_size < size):
            return np.float('nan')
        else:
            return np.average(self.ask_prices, weights=int_sizes)

    def getPriceAtLevel(self, level, is_bid):
        if is_bid:
            return self.bid_prices[level]
        else:
            return self.ask_prices[level]

    def getVolumeAhead(self, price, is_bid):
        volume_ahead = np.float(0.0)
        for i in np.arange(self.n_levels):
            if is_bid:
                if price < self.bid_prices[i]:
                    volume_ahead += self.bid_quantity[i]
            else:
                if price > self.ask_prices[i]:
                    volume_ahead += self.ask_quantity[i]

        return volume_ahead

    def getTOBWMID(self):
        return ((self.bid_prices[0] * self.ask_quantity[0]) + (self.ask_prices[0] * self.bid_quantity[0])) / (
                self.bid_quantity[0] + self.ask_quantity[0])

    def getOBWMID(self):
        self.bid_vwap = np.average(self.bid_prices, weights=self.bid_quantity)
        self.ask_vwap = np.average(self.ask_prices, weights=self.ask_quantity)
        return ((self.bid_vwap * self.total_ask_size) +
                (self.ask_vwap * self.total_bid_size)) / (
                       self.total_bid_size + self.total_ask_size)

    def getVWAPWMID(self, volume):
        # Calculate weighted mid part 2
        bid_vwap = self.c_get_bid_vwap(volume, 0)
        ask_vwap = self.c_get_ask_vwap(volume, 0)
        return ((bid_vwap * volume) + (ask_vwap * volume)) / (volume * 2.0)


class CustomGateIoOrderBook(CustomOrderBook):

    def __init__(self):
        super().__init__()

    def extractOB(self):
        bids = iter(self.ob_dict['book']['bid'].items())
        asks = iter(self.ob_dict['book']['ask'].items())
        for i in range(self.n_levels):
            bid_price, bid_quantity = next(bids)
            ask_price, ask_quantity = next(asks)
            self.bid_prices[i] = np.float(bid_price)
            self.bid_quantity[i] = np.float(bid_quantity)
            self.ask_prices[i] = np.float(ask_price)
            self.ask_quantity[i] = np.float(ask_quantity)

        # total bid & ask sizes
        self.total_bid_size = np.sum(self.bid_quantity)
        self.total_ask_size = np.sum(self.ask_quantity)
        self.bid_price = self.bid_prices[0]
        self.ask_price = self.ask_prices[0]

    def extractTimestampSequence(self):
        self.exchange_timestamp = self.ob_dict['timestamp']
        self.receipt_timestamp = self.ob_dict['receipt_timestamp']
        if 'sequence_number' in self.optional_features:
            self.sequence_number = self.ob_dict[self.sequence_number_feature_name]
