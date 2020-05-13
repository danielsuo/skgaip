import jax.numpy as np
import jax

from timecast.learners import BaseLearner
from timecast.optim import SGD

from ealstm.gaip.utils import BatchedMeanSquareError


class ARStateless(BaseLearner):
    def __init__(
        self, input_dim: int, output_dim: int, window_size: int, optimizer=None, loss=None
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._window_size = window_size
        self._optimizer = optimizer or SGD()
        self._loss = loss or BatchedMeanSquareError()

        W = np.zeros((window_size * input_dim + 1, output_dim))
        self._params = {"W": W}

        def _predict(params, x):
            x = x[np.newaxis, :]
            X = np.hstack((np.ones((x.shape[0], 1)), x.reshape(x.shape[0], -1))).reshape(
                x.shape[0], -1, 1
            )
            return np.tensordot(X, params["W"], axes=(1, 0))

        self._predict_jit = jax.jit(lambda params, X: _predict(params, X))
        self._grad = jax.jit(
            jax.grad(lambda params, X, y: self._loss.compute(_predict(params, X), y))
        )
        self._value_and_grad = jax.jit(
            jax.value_and_grad(lambda params, X, y: self._loss.compute(_predict(params, X), y))
        )

        def _predict_and_update(params, xy):
            value = self._predict_jit(params, xy[0])
            gradients = self._grad(params, xy[0], xy[1])
            params = self._optimizer.update(params, gradients)
            return params, value

        self._predict_and_update_jit = jax.jit(lambda params, xy: _predict_and_update(params, xy))

    def predict(self, X):
        return jax.vmap(self._predict_jit, in_axes=(None, 0))(self._params, X).reshape(-1, 1)

    def update(self, X, y):
        gradients = jax.vmap(self._grad, in_axes=({"W": None, "b": None}, 0, 0), out_axes=0)(
            self._params, X, y
        )
        gradients["W"] = gradients["W"].mean(axis=0)
        gradients["b"] = gradients["b"].mean(axis=0)
        self._params = self._optimizer.update(self._params, gradients)

    def predict_and_update(self, X, y):
        # TODO: think about batching
        self._params, value = jax.lax.scan(self._predict_and_update_jit, self._params, (X, y))
        return value.reshape(-1, 1)
