import json
from pathlib import Path
import jax.numpy as np

from ealstm.main import GLOBAL_SETTINGS


def load_cfg(cfg_path):
    cfg = json.load(open(cfg_path, "r"))
    cfg["camels_root"] = Path(cfg["camels_root"])
    cfg["run_dir"] = Path(cfg["run_dir"])
    cfg.update(GLOBAL_SETTINGS)
    return cfg


def MSE(x, y):
    return ((x - y) ** 2).mean()


def NSE(y_true, y_pred):
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


def BMSE(y_pred: np.ndarray, y_true: np.ndarray):
    return np.mean(np.mean((y_pred - y_true) ** 2, axis=tuple(range(1, y_true.ndim))))


def batch_window(X, window_size, offset=0):
    num_windows = X.shape[0] - window_size + 1
    return np.swapaxes(
        np.stack([np.roll(X, shift=-(i + offset), axis=0) for i in range(window_size)]), 0, 1
    )[:num_windows]
