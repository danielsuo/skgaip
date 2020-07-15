import os
import pickle

import numpy as np
from keras.models import load_model, Model
from timecast.learners import AR, PredictLast, Transform, Residual
from timecast.utils.losses import MeanSquareError

from fusion_data import FusionData
from utils import experiment

ex = experiment("train_ar_residual")


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
    freeze,
):
    # with ex.open_resource(os.path.join(data_dir, shot_data), "rb") as data_file:
    # with ex.open_resource(os.path.join(data_dir, data_keys), "rb") as keys_file:
    # with ex.open_resource(os.path.join(data_dir, model_path), "rb") as model_file:
    # data = FusionData(
    # data_file,
    # keys_file,
    # input_dim=input_dim,
    # input_index=input_index,
    # output_dim=output_dim,
    # output_index=output_index,
    # warning=warning,
    # filter_size=filter_size,
    # batch_size=batch_size,
    # normalize=normalize,
    # model_path=os.path.join(data_dir, model_path),
    # )
    # pl = PredictLast(input_dim=1, output_dim=1)
    # ar = AR(
    # input_dim=data.input_dim,
    # output_dim=output_dim,
    # window_size=window_size,
    # fit_intercept=fit_intercept,
    # constrain=constrain,
    # normalize=False,
    # )

    # if freeze:
    # ar.freeze(params=True, stats=True)

    # pl_t = Transform(pl, input_dim=200, transformed_input_dim=1, output_dim=1, transformed_output_dim=1, X_transform=lambda X: X[:, 3])
    # learner = Residual([pl_t, ar])
    # learner.fit(data.featurize(), alpha=1)

    # path = os.path.join(ex.observers[1].dir, result_path)
    # print("Saving model to {}".format(path))
    # learner.save(path)
    data = FusionData(
        open("FRNN_1d_sample/shot_data.npz", "rb"),
        open("FRNN_1d_sample/train_list.npy", "rb"),
        input_dim=142,
        input_index=3,
        output_dim=1,
        output_index=3,
        window_size=1,
        filter_size=128,
        batch_size=128,
        warning=30,
        model_path="FRNN_1d_sample/FRNN_1D_sample.h5",
    )

    ar3 = AR(
        input_dim=200,
        output_dim=1,
        window_size=1,
        fit_intercept=True,
        constrain=False,
        normalize=False,
    )
    ar3.fit(
        [
            (
                data.masked_inputs,
                data.masked_outputs.reshape(data.masked_outputs.shape[0], 1) -
                data.original_inputs[:, 3].reshape(data.original_inputs.shape[0], 1),
                None,
            )
        ]
    )

    del data
    data = FusionData(
        open("FRNN_1d_sample/shot_data.npz", "rb"),
        open("FRNN_1d_sample/test_list.npy", "rb"),
        input_dim=142,
        input_index=3,
        output_dim=1,
        output_index=3,
        window_size=1,
        filter_size=128,
        batch_size=128,
        warning=30,
        model_path="FRNN_1d_sample/FRNN_1D_sample.h5",
    )

    pred = {}
    true = {}
    mse = {}
    index = 0
    for i, (X, y, shot) in enumerate(data.featurize()):
        print("Processing shot {}: {} ({}, {})".format(i, shot, X.shape, y.shape))
        pred[shot] = ar3.predict(X) + data.original_inputs[index : index + X.shape[0], 3].reshape(
            X.shape[0], 1
        )
        true[shot] = y
        mse[shot] = MeanSquareError().compute(pred[shot], true[shot])
        index += X.shape[0]

    del data
    path = os.path.join(ex.observers[1].dir, "results.pkl")
    print("Saving to {}".format(path))
    pickle.dump({"pred": pred, "true": true, "mse": mse}, open(path, "wb"))
