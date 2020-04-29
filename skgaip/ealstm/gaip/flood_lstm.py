import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from timecast.learners import BaseLearner

from ealstm.main import DEVICE
from ealstm.main import evaluate_basin
from ealstm.main import Model
from ealstm.papercode.datautils import reshape_data

from ealstm.gaip.utils import load_cfg

import numpy as np


class FloodLSTM(BaseLearner):
    def __init__(self, cfg_path, input_dim=5, output_dim=1):
        self._input_dim = (input_dim,)
        self._output_dim = output_dim

        self.cfg = load_cfg(cfg_path)
        self.model = Model(
            input_size_dyn=(5 if (self.cfg["no_static"] or not self.cfg["concat_static"]) else 32),
            input_size_stat=(0 if self.cfg["no_static"] else 27),
            hidden_size=self.cfg["hidden_size"],
            dropout=self.cfg["dropout"],
            concat_static=self.cfg["concat_static"],
            no_static=self.cfg["no_static"],
        ).to(DEVICE)

        weight_file = os.path.join(self.cfg["run_dir"], "model_epoch30.pt")
        self.model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    def predict(self, X):
        """Assumes we get one basin's data at a time
        """
        X = np.asarray(X)
        y = np.ones((X.shape[0], 1))
        X, y = reshape_data(X, y, self.cfg["seq_length"])

        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        loader = DataLoader(TensorDataset(X, y), batch_size=1024, shuffle=False)
        preds, obs = evaluate_basin(self.model, loader)
        return preds

    def update(self, X, y, **kwargs):
        pass
