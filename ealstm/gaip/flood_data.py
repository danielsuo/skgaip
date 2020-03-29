import os
import json
from pathlib import Path
import jax.numpy as np

from ealstm.main import GLOBAL_SETTINGS
from ealstm.main import get_basin_list
from ealstm.main import load_attributes
from ealstm.papercode.datasets import CamelsTXT

from ealstm.gaip.utils import load_cfg


class FloodData:
    def __init__(self, cfg_path):
        self.cfg = load_cfg(cfg_path)
        self.basins = get_basin_list()
        self.db_path = os.path.join(self.cfg["run_dir"], "attributes.db")
        self.attributes = load_attributes(
            db_path=self.db_path, basins=self.basins, drop_lat_lon=True
        )

    def generator(self, is_train=False, with_attributes=True):
        for basin in self.basins:
            ds_test = CamelsTXT(
                camels_root=self.cfg["camels_root"],
                basin=basin,
                dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                is_train=is_train,
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
            yield X, ds_test.y, basin
