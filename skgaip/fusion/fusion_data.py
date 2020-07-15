import binpacking
import numpy as np
from keras.models import load_model, Model


class FusionData:
    def __init__(
        self,
        data,
        keys,
        input_dim=142,
        input_index=3,
        output_dim=1,
        output_index=3,
        warning=30,
        filter_size=128,
        batch_size=128,
        normalize=True,
        headless=True,
        model_path=None,
        **kwargs
    ):
        print("Loading data from {}".format(data))
        data = np.load(data, allow_pickle=True)["shot_data"].item()
        print("Found {} shots".format(len(data.keys())))

        print("Loading keys from {}".format(keys))
        self.keys = np.load(keys)
        print("Found {} keys".format(len(self.keys)))

        print("Filtering data")
        self.disruptive = {key: data[key]["is_disruptive"] for key in self.keys}
        self.data = {key: data[key]["X"] for key in self.keys}

        print("Filtered to {} shots".format(len(self.data.keys())))

        if normalize:
            print("Normalizing data")
            self.normalize()

        print("Packing data")
        self.pack(
            input_dim=input_dim,
            input_index=input_index,
            output_dim=output_dim,
            output_index=output_index,
            warning=warning,
            filter_size=filter_size,
            batch_size=batch_size,
        )
        if model_path is not None:
            print("Loading model")
            # Open just to record in sacred
            model = load_model(model_path)

            if headless:
                # Remove final layer (disruptivity score decoder)
                model = Model(inputs=model.input, outputs=model.get_layer("lstm_2").output)

            print("Transforming data according to {}".format(model))
            self.model = model
            self.transform()

        print("Removing pad")
        self.remove_pad()

    # Lol, very inefficient
    def normalize(self):
        num_measurements = np.sum([shot.shape[0] for shot in self.data.values()])
        data = np.zeros((num_measurements, self.data[self.keys[0]].shape[1]))

        index = 0
        for key, shot in self.data.items():
            data[index : index + shot.shape[0], :] = shot
            index += shot.shape[0]

        mean = data.mean(axis=0)
        std = data.std(axis=0)

        self.data = {key: ((self.data[key] - mean) / std) for key in self.keys}

    def pack(
        self,
        input_dim=1,
        input_index=3,
        output_dim=1,
        output_index=3,
        warning=30,
        filter_size=1,
        batch_size=1,
    ):
        self.warning = warning
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.input_dim = 1 if input_dim == 1 else self.data[list(self.data.keys())[0]].shape[1]
        self.input_index = input_index
        self.output_dim = 1 if output_dim == 1 else self.input_dim
        self.output_index = output_index

        balls = {
            key: (self.data[key].shape[0] - self.warning) // self.filter_size for key in self.keys
        }

        self.packing = binpacking.to_constant_bin_number(balls, self.batch_size)

        # Find batch with most chunks and multiply by filter_size to get total
        # time steps
        time_steps = self.filter_size * np.max(
            [np.sum(list(pack.values())) for pack in self.packing]
        )

        # Create one input per batch of size time_steps by number of features
        inputs = np.zeros((self.batch_size, time_steps, self.input_dim))

        # Create one output per batch of size time_steps by output dimensions
        outputs = np.zeros((self.batch_size, time_steps, self.output_dim))

        self.start = {}
        self.end = {}

        # Loop over each batch
        for batch_id, pack in enumerate(self.packing):
            # Maintain an array index into input/output because shots are of
            # different lengths
            index = 0

            # Loop over all time series assigned to a batch
            for shot_index, num_chunks in pack.items():

                # Shot length in time steps
                shot_length = num_chunks * self.filter_size

                # Inputs ends warning steps before the end of data
                end = self.data[shot_index].shape[0] - self.warning

                # Start truncates so that (end - start) % filter_size == 0
                start = end % self.filter_size

                self.start[shot_index] = start
                self.end[shot_index] = end

                # Capture time series from truncated start through end minus
                # `warning` steps
                if self.input_dim > 1:
                    inputs[batch_id, index : (index + shot_length), :] = self.data[shot_index][
                        start:end, :
                    ]
                else:
                    inputs[batch_id, index : (index + shot_length), :] = self.data[shot_index][
                        start:end, self.input_index
                    ].reshape(-1, 1)

                # Capture corresponding output time series (shift by `warning`
                # steps)
                if self.output_dim > 1:
                    outputs[batch_id, index : (index + shot_length), :] = self.data[shot_index][
                        start + self.warning :, :
                    ]
                else:
                    outputs[batch_id, index : (index + shot_length), :] = self.data[shot_index][
                        start + self.warning :, self.output_index
                    ].reshape(-1, 1)

                # Update index
                index += shot_length

        self.inputs = inputs
        self.outputs = outputs
        self.mask = self.inputs.any(axis=2)

    def transform(self):
        results = np.zeros(
            (self.inputs.shape[0], self.inputs.shape[1], self.model.output.shape[-1])
        )

        # Loop through each window and pass into the LSTM for results
        for filter_index in range(self.inputs.shape[1] // self.filter_size):
            start = filter_index * self.filter_size
            end = start + self.filter_size
            results[:, start:end, :] = self.model.predict(
                self.inputs[:, start:end, :], batch_size=self.batch_size,
            )

        self.original_inputs = self.inputs
        self.inputs = results
        self.input_dim = int(self.model.output.shape[-1])

    # Lol I can't call this mask...
    def remove_pad(self):
        # Remove any padding created during pack to be filter_size-aligned
        self.masked_outputs = self.outputs[self.mask]
        self.masked_inputs = self.inputs[self.mask]
        self.original_inputs = self.original_inputs[self.mask]
        self.masked_packing = [y for pack in self.packing for y in pack.items()]

    def featurize(self, window_size=1, yield_size=1, message=False):
        # TODO: Ignores arguments

        index = 0
        for shot_index, num_chunks in self.masked_packing:
            start = index
            end = start + num_chunks * self.filter_size

            yield self.masked_inputs[start:end, :], self.masked_outputs[
                start:end, :
            ], shot_index

            index = end

        # X = np.zeros((yield_size, window_size * self.inputs.shape[-1]))
        # Y = np.zeros((yield_size, self.outputs.shape[-1]))
        # yield_counter = 0
        # shot_counter = 0

        # for batch_id, pack in tqdm(
        # enumerate(self.packing), total=self.batch_size, desc="batch", position=0
        # ):
        # index = 0
        # for shot_index, num_chunks in tqdm(
        # pack.items(), total=len(pack.items()), desc="shot", position=1, leave=None,
        # ):
        # shot_counter += 1
        # shot_length = num_chunks * self.filter_size

        # for time_index in tqdm(
        # range(window_size, shot_length + 1), desc="time", position=2, leave=None,
        # ):
        # yield_index = yield_counter % yield_size
        # if yield_index == 0:
        # if X is not None:
        # yield X, Y, shot_index
        # X[:, :], Y[:, :] = 0, 0

        # start = index + time_index - window_size
        # end = index + time_index
        # if X.shape[1] > 1:
        # X[yield_index] = self.inputs[batch_id, start:end, :].ravel()
        # else:
        # X[yield_index] = self.inputs[batch_id, start:end]

        # if Y.shape[1] > 1:
        # Y[yield_index] = self.outputs[batch_id, start:end, :].ravel()
        # else:
        # Y[yield_index] = self.outputs[batch_id, start:end]
        # yield_counter += 1
        # index += shot_length

    def __getitem__(self, key):
        return self.data[key]
