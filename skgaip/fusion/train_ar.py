import os
import pickle

import numpy as np
from keras.models import load_model, Model
from timecast.learners import AR

from fusion_data import FusionData
from utils import experiment

ex = experiment("train_ar")


@ex.config
def config():
    data_dir = "FRNN_1d_sample"
    data_keys = "train_list.npy"
    shot_data = "shot_data.npz"
    input_dim = 142
    input_index = 3
    output_dim = 1
    output_index = 3
    window_size = 1
    filter_size = 128
    batch_size = 128
    warning = 30
    model_path = "FRNN_1D_sample.h5"
    result_path = "results.pkl"
    fit_intercept = True
    constrain = False
    normalize = True
    freeze = True


@ex.automain
def main(
    data_dir,
    data_keys,
    shot_data,
    input_dim,
    input_index,
    output_dim,
    output_index,
    window_size,
    filter_size,
    batch_size,
    warning,
    model_path,
    result_path,
    fit_intercept,
    constrain,
    normalize,
    freeze
):
    with ex.open_resource(os.path.join(data_dir, shot_data), "rb") as data_file:
        with ex.open_resource(os.path.join(data_dir, data_keys), "rb") as keys_file:
            with ex.open_resource(os.path.join(data_dir, model_path), "rb") as model_file:
                data = FusionData(
                    data_file,
                    keys_file,
                    input_dim=input_dim,
                    input_index=input_index,
                    output_dim=output_dim,
                    output_index=output_index,
                    warning=warning,
                    filter_size=filter_size,
                    batch_size=batch_size,
                    normalize=normalize,
                    model_path=os.path.join(data_dir, model_path),
                )

    learner = AR(
        input_dim=data.input_dim,
        output_dim=output_dim,
        window_size=window_size,
        fit_intercept=fit_intercept,
        constrain=constrain,
        # Data loader will handle normalization
        normalize=False,
    )
    learner.fit(data.featurize(), alpha=1)

    if freeze:
        learner.freeze(params=True, stats=True)

    learner.reset()

    path = os.path.join(ex.observers[1].dir, result_path)
    print("Saving model to {}".format(path))
    learner.save(path)
