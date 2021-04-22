from __future__ import division, print_function

import datetime

import matplotlib.pyplot as plt
import numpy as np
from data_reader import DataReader
from scipy import signal
from sklearn.metrics import mean_squared_error


class CGMSData(object):
    """Data set"""

    def __init__(self, fmt, filepath, sampling_interval):
        self.interval = datetime.timedelta(minutes=sampling_interval)
        reader = DataReader(fmt, filepath, self.interval)
        self.raw_data = reader.read()
        self.data = list(self.raw_data)
        print(f"Reading {len(self.data)} segments")

        self.sampling_horizon, self.prediction_horizon = 0, 0
        self.scale, self.train_test_ratio = 0, 0
        self.n, self.set_cutpoint = len(self.data), False
        self.train_x, self.train_y, self.train_weights = None, None, None
        self.test_x, self.test_y = None, None
        self.train_n, self.test_n = 0, 0
        self.train_idx = None

    def _smooth(self, window_length, polyorder):
        self.data = map(
            lambda x: signal.savgol_filter(x, window_length, polyorder),
            filter(lambda x: x.size > window_length, self.raw_data),
        )

    def _cut_point(self):
        s = list(map(lambda d: d.size, self.data))
        s = np.cumsum(s)
        if np.isinf(self.train_test_ratio):
            c = s[-1]
        else:
            c = s[-1] * self.train_test_ratio / (1 + self.train_test_ratio)
        return max(np.searchsorted(s, c, side="right"), 1)

    def _build_dataset(self, beg, end, padding):
        print(f"Requesting data from {beg} to {end}")
        x, y = [], []
        l = self.sampling_horizon + self.prediction_horizon
        for d in self.data[beg:end]:
            d = np.array(d)
            for i in range(
                d.size - self.sampling_horizon - self.prediction_horizon + 1
            ):
                x.append(d[i : (i + self.sampling_horizon)])
                if padding == "History":
                    y.append(d[(i + self.sampling_horizon) : (i + l)])
                else:
                    y.append(d[i + l - 1])
        if padding == "None" or padding == "History":
            return np.array(x), np.array(y)
        if padding == "Same":
            return np.array(x), np.tile(y, [self.sampling_horizon, 1]).T
        raise ValueError("Unsupported padding " + padding)

    def _scale(self, ohio_data):
        if not ohio_data:
            self.train_x *= self.scale
            self.train_y *= self.scale
            self.test_x *= self.scale
            self.test_y *= self.scale
        else:
            # mean and std of training data of OhioT1DM, Bevan
            mean = 158.288
            std = 60.565
            self.train_x = (self.train_x - mean) / std
            self.train_y = (self.train_y - mean) / std
            self.test_x = (self.test_x - mean) / std
            self.test_y = (self.test_y - mean) / std

    def reset(
        self,
        sampling_horizon,
        prediction_horizon,
        scale,
        train_test_ratio,
        smooth,
        padding,
        target_weight,
    ):
        self.sampling_horizon = sampling_horizon
        self.prediction_horizon = prediction_horizon
        self.scale = scale
        self.train_test_ratio = train_test_ratio

        if smooth:
            window_length = sampling_horizon
            if window_length % 2 == 0:
                window_length += 1
            self._smooth(window_length, window_length - 4)
        print("# time series: {}".format(len(self.data)))
        c = self._cut_point()
        self.train_x, self.train_y = self._build_dataset(0, c, padding)
        self.test_x, self.test_y = self._build_dataset(c, len(self.data), padding)
        self.train_n = self.train_x.shape[0]
        self.test_n = self.test_x.shape[0]
        print("Train data size: %d" % self.train_n)
        print("Test data size: %d" % self.test_n)
        self._scale()

        self.train_weights = None
        if padding != "None":
            l = self.train_y.shape[1]
            self.train_weights = np.full(l, (1 - target_weight) / (l - 1))
            self.train_weights[-1] = target_weight

        self.train_idx = np.random.permutation(self.train_n)

    def t0_baseline(self):
        y = self.test_y
        if y.ndim == 2:
            y = y[:, -1]
        return mean_squared_error(y, self.test_x[:, -1]) ** 0.5 / self.scale

    def train_next_batch(self, batch_size):
        if self.train_idx.size < batch_size:
            self.train_idx = np.random.permutation(self.train_n)
        idx = self.train_idx[:batch_size]
        self.train_idx = self.train_idx[batch_size:]
        return self.train_x[idx], self.train_y[idx], self.train_weights

    def test(self):
        weights = None
        if self.train_weights is not None:
            weights = np.zeros_like(self.train_y[0])
            weights[-1] = 1
        return self.test_x, self.test_y, weights

    def render_data(self, n=3):
        plt.figure()
        for d in self.data[:n]:
            plt.plot(d)
        plt.xlabel("Time (%d min)" % (self.interval.total_seconds() / 60))

        dist_l2 = 0.04
        x0 = None
        n = 0
        for x in self.train_x:
            if np.var(x) < 0.001:
                continue
            l = np.linalg.norm(self.train_x - x, axis=1)
            idx = np.nonzero(l < dist_l2)[0]
            if idx.size > n:
                n = idx.size
                x0 = x
        l = np.linalg.norm(self.train_x - x0, axis=1)
        idx = np.nonzero(l < dist_l2)[0]
        plt.figure()
        for i in idx:
            plt.plot(self.train_x[i] / self.scale)
        y = self.train_y[idx] if self.train_y.ndim == 1 else self.train_y[idx, -1]
        plt.plot(
            np.full(n, self.sampling_horizon + self.prediction_horizon - 1),
            y / self.scale,
            "o",
        )
        plt.xlabel("Time (%d min)" % (self.interval.total_seconds() / 60))
        plt.title("%d samples" % n)
        plt.show()

    def test_patient(self, ptid=-1):
        x = []
        while self.data[ptid].size < (self.sampling_horizon + self.prediction_horizon):
            ptid -= 1
        d = self.data[ptid]
        for i in range(d.size - self.sampling_horizon - self.prediction_horizon + 1):
            x.append(d[i : (i + self.sampling_horizon)])
        return ptid, np.array(x) * self.scale

    def render_prediction(self, ptid, y, yerr=None, show=False):
        plt.figure()
        plt.plot(self.data[ptid], "bo-", label="Truth")
        x = np.arange(y.size) + (self.sampling_horizon + self.prediction_horizon - 1)
        if yerr is not None:
            plt.errorbar(
                x, y / self.scale, yerr=yerr / self.scale, fmt="none", ecolor="grey"
            )
        plt.plot(x, y / self.scale, "gv-", label="Prediction")
        plt.legend(loc="best")
        plt.xlabel("Time (%d min)" % (self.interval.total_seconds() / 60))
        if show:
            plt.show()
        else:
            plt.savefig("prediction_%d.png" % (ptid + len(self.data)))
