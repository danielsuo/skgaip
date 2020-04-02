import sys
import pickle

import jax
import jax.numpy as np
from flax import nn

from ealstm.gaip import FloodData
from ealstm.gaip.utils import MSE, NSE

from timecast.optim import RMSProp
from timecast.learners import AR
from timecast.objectives import map_grad, residual

if np.ones(1).dtype != np.dtype("float64"):
    print("FAIL!!!!")
    exit(1)

cfg_path = "/home/dsuo/src/toy_flood/ealstm/runs/run_2503_0429_seed283956/cfg.json"
ea_data = pickle.load(open("../ealstm/runs/run_2503_0429_seed283956/lstm_seed283956.p", "rb"))
flood_data = FloodData(cfg_path)

LR = 10 ** float(sys.argv[1])
AR_INPUT_DIM = 32
AR_OUTPUT_DIM = 1
BETA = float(sys.argv[2])

results = {}
mses = []
nses = []


class Residual(nn.Module):
    def apply(self, x, output_features, history_len, history):
        y_ar = AR(x=x[1], output_features=output_features, history_len=history_len, history=history)
        return (x[0], y_ar)


for X, _, basin in flood_data.generator():

    with nn.stateful() as state:
        model_def = Residual.partial(
            output_features=1, history_len=270, history=X[: flood_data.cfg["seq_length"] - 1]
        )
        ys, params = model_def.init_by_shape(jax.random.PRNGKey(0), [(1, 32)])
        model = nn.Model(model_def, params)
    optim_def = RMSProp(learning_rate=LR, beta_2=BETA)
    optimizer = optim_def.create(model)

    X = X[flood_data.cfg["seq_length"] - 1 :]
    Y_lstm = np.array(ea_data[basin].qsim).reshape(-1, 1)
    Y = np.array(ea_data[basin].qobs).reshape(-1, 1)

    Y_hat = map_grad(
        (Y_lstm, X), Y, optimizer, state, residual, MSE
    )

    mse = MSE(Y, Y_hat)
    nse = NSE(Y, Y_hat)
    results[basin] = {
        "mse": mse,
        "nse": nse,
        "count": X.shape[0],
        "avg_mse": np.mean(np.array(mses)),
        "avg_nse": np.mean(np.array(nses)),
    }
    mses.append(mse)
    nses.append(nse)
    print(basin, mse, nse, X.shape[0], np.mean(np.array(mses)), np.mean(np.array(nses)))

with open("beta_{}_{}.pkl".format(sys.argv[1], sys.argv[2]), "wb") as f:
    pickle.dump(results, f)
