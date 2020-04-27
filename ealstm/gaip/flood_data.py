import os
import jax.numpy as np

from ealstm.main import GLOBAL_SETTINGS
from ealstm.main import get_basin_list
from ealstm.main import load_attributes
from ealstm.papercode.datasets import CamelsTXT

from ealstm.gaip.utils import load_cfg


class FloodData:
    def __init__(self, cfg_path, is_train=False, with_attributes=True):
        self.cfg = load_cfg(cfg_path)
        self.basins = get_basin_list()
        self.db_path = os.path.join(self.cfg["run_dir"], "attributes.db")
        self.attributes = load_attributes(
            db_path=self.db_path, basins=self.basins, drop_lat_lon=True
        )
        self.is_train = is_train
        self.with_attributes = with_attributes
        self.start = (
            GLOBAL_SETTINGS["train_start"] if self.is_train else GLOBAL_SETTINGS["val_start"]
        )
        self.end = GLOBAL_SETTINGS["train_end"] if self.is_train else GLOBAL_SETTINGS["val_end"]

    def generator(self):
        for basin in self.basins:
            yield self[basin] + (basin,)

    def __getitem__(self, basin: str):
        ds_test = CamelsTXT(
            camels_root=self.cfg["camels_root"],
            basin=basin,
            dates=[self.start, self.end],
            is_train=self.is_train,
            seq_length=self.cfg["seq_length"],
            with_attributes=True,
            attribute_means=self.attributes.mean(),
            attribute_stds=self.attributes.std(),
            concat_static=self.cfg["concat_static"],
            db_path=self.db_path,
            reshape=False,
            torchify=False,
        )
        X = np.concatenate(
            (ds_test.x, np.tile(np.array(ds_test.attributes), (ds_test.x.shape[0], 1))), axis=1
        )
        return X, ds_test.y
