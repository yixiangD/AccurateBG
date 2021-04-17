import numpy as np


class MixUp:
    def __init__(self, train_x, train_y, minor_x, minor_y, major_x, major_y, alpha, m):
        """
        alpha: float
            hyperparameter for Beta distribution
        m: integer (> 1)
            the multiplication factor indicating the copies of minority
            class existing in the training data
        """
        self.train_x = train_x
        self.train_y = train_y
        self.minor_x = minor_x
        self.minor_y = minor_y
        self.major_x = major_x
        self.major_y = major_y
        self.alpha = alpha
        self.m = m

    def mixup_by(self, option):
        assert self.m >= 2
        if option == "unbiased":
            return self._mixup(self.train_x, self.train_y)
        elif option == "minority":
            return self._mixup(self.minor_x, self.minor_y)
        elif option == "inner":
            new_major_x, new_major_y = self._mixup(self.major_x, self.major_y)
            new_minor_x, new_minor_y = self._mixup(self.minor_x, self.minor_y)
            return np.vstack((new_major_x, new_minor_x)), np.vstack(
                (new_major_y, new_minor_y)
            )
        else:
            raise NotImplementedError("{} not implemented".format(option))

    def _mixup(self, data_x, data_y):
        num_data, indata_dim = data_x.shape[0], data_x.shape[1]
        if len(data_y.shape) == 1:
            data_y = data_y[:, None]
        data = np.hstack((data_x, data_y))
        new_data = np.repeat(data, self.m, axis=0)
        lmbda = np.random.beta(self.alpha, self.alpha, size=num_data * self.m)[:, None]
        new_data = lmbda * new_data + (1 - lmbda) * new_data
        return new_data[:, :indata_dim], new_data[:, indata_dim:]
