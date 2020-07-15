import os
import time
import pickle

import numpy as np
from timecast.learners import AR, PredictLast, PredictConstant
from timecast.utils.losses import MeanSquareError
from timecast import load_learner

from fusion_data import FusionData
from utils import experiment

ex = experiment("baseline")


@ex.config
def config():
    learner_type = "PredictLast"
    learner_path = None
    data_dir = "FRNN_1d_sample"
    data_keys = "test_list.npy"
    shot_data = "shot_data.npz"
    input_dim = 1
    input_index = 3
    output_dim = 1
    output_index = 3
    window_size = 1
    filter_size = 1
    batch_size = 1
    warning = 30
    model_path = None
    result_path = "results.pkl"
    fit_intercept = False
    constrain = False
    normalize = True


@ex.named_config
def config_ar():
    learner_type = "AR"
    learner_path = "sacred/experiments/train_ar/89/results.pkl"
    data_dir = "FRNN_1d_sample"
    data_keys = "test_list.npy"
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


@ex.automain
def main(
    learner_type,
    learner_path,
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

    if learner_type == "PredictLast":
        learner_type = PredictLast
    elif learner_type == "PredictConstant":
        learner_type = PredictConstant
    elif learner_type == "AR":
        learner_type = AR

    if learner_path is None:
        learner = learner_type(
            # Input dimension may have gotten transformed
            input_dim=int(data.input_dim),
            output_dim=output_dim,
            window_size=window_size,
            fit_intercept=fit_intercept,
            constrain=constrain,
            # Data loader will handle normalization
            normalize=False,
        )
    else:
        learner = load_learner(learner_path)

    print(learner.to_dict())

    pred = {}
    true = {}
    mse = {}

    for i, (X, y, shot) in enumerate(data.featurize()):
        print("Processing shot {}: {} ({}, {})".format(i, shot, X.shape, y.shape))
        # if shot not in pred:
        start = time.time()
        pred[shot] = learner.predict(X)
        print("  ...took {} seconds".format(time.time() - start))
        true[shot] = y
        learner.update(X, y)
        mse[shot] = MeanSquareError().compute(pred[shot], true[shot])

        learner.reset()

    path = os.path.join(ex.observers[1].dir, result_path)
    print("Saving data to {}".format(path))
    pickle.dump({"pred": pred, "true": true, "mse": mse}, open(path, "wb"))
