import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def PMSE(pred, true):
    pred = pred[..., 0]
    true = true[..., 0]
    left = true[:, :-2]
    middle = true[:, 1:-1]
    right = true[:, 2:]
    mean = true.mean(axis=1, keepdims=True)
    peaks = (middle > left) & (middle > right) & (middle > mean)
    mse = np.mean(np.square(true[:, 1:-1][peaks] - pred[:, 1:-1][peaks]))
    return mse


def PMAE(pred, true):
    pred = pred[..., 0]
    true = true[..., 0]
    left = true[:, :-2]
    middle = true[:, 1:-1]
    right = true[:, 2:]
    mean = true.mean(axis=1, keepdims=True)
    peaks = (middle > left) & (middle > right) & (middle > mean)
    mae = np.mean(np.abs(true[:, 1:-1][peaks] - pred[:, 1:-1][peaks]))
    return mae


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    pmse = PMSE(pred, true)
    pmae = PMAE(pred, true)

    return mae, mse, rmse, mape, mspe, pmse, pmae
