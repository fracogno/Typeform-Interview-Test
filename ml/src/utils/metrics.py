import numpy as np


def regression_error(pred, labels):
    error = np.abs(pred - labels)
    return round(np.mean(error) * 100., 3)
