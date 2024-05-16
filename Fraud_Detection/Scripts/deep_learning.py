import math
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)

from keras import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from sklearn.metrics import accuracy_score

class BuildNeuralNetArchitecture(object):

    def __init__(self,
                 n_inputs=None,
                 output_layer_nodes=None,
                 output_layer_activation=None,
                 activation="relu",
                 n_dense_hidden_layers=2,
                 first_dense_hidden_layer_nodes=500,
                 last_dense_hidden_layer_nodes=100,
                 negative_node_incrementation=True,
                 add_lstm_layers=False,
                 look_back=3,
                 n_lstm_hidden_layers=1,
                 first_lstm_layer_nodes=100,
                 last_lstm_layer_notes=25,
                 use_bidirectional_layers=True,
                 loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy']):
        """
        n_dense_hidden_layers: int
            number of hidden layers in model
            To be clear, this excludes the input and output layer.

        first_dense_hidden_layer_nodes: int
            Number of nodes in the first hidden layer

        last_dense_hidden_layer_nodes: int
            Number of nodes in the last hidden layer (this is the layer just prior to the output layer)

         activation: string
             Name of activation function to use in hidden layers (this excludes the output layler)
        """

        # dense layer parameters
        self.n_dense_hidden_layers = n_dense_hidden_layers
        self.n_inputs = n_inputs
        self.first_dense_hidden_layer_nodes = first_dense_hidden_layer_nodes
        self.last_dense_hidden_layer_nodes = last_dense_hidden_layer_nodes
        self.activation = activation
        self.negative_node_incrementation = negative_node_incrementation
        self.output_layer_nodes = output_layer_nodes
        self.output_layer_activation = output_layer_activation

        # lstm layer parameters
        self.add_lstm_layers = add_lstm_layers
        self.look_back = look_back
        self.n_lstm_hidden_layers = n_lstm_hidden_layers
        self.first_lstm_layer_nodes = first_lstm_layer_nodes
        self.last_lstm_layer_notes = last_lstm_layer_notes
        self.input_shape = (self.look_back, self.n_inputs)
        self.use_bidirectional_layers = use_bidirectional_layers

        # complie method parameters
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        # will contain list of nodes for each layer for each layer type
        # i.e, key: "dense", value: [500, 400, 300, 200, 100]
        # i.e, key: "lstm", value: [100, 50]
        self.n_nodes = defaultdict(list)


    def __call__(self):
        return self.create_dense_model()

    def gen_layer_nodes(self, layer_type):
        """
        Generates and returns the number of nodes in each hidden layer.
        To be clear, this excludes the input and output layer.
        """

        # throws an error if n_dense_hidden_layers is less than 2
        # assert self.n_layers >= 2, "n_dense_hidden_layers must be 2 or greater"

        if layer_type == "dense" and self.n_dense_hidden_layers > 0:
            if self.n_dense_hidden_layers == 1:
                self.n_nodes[layer_type].append(self.first_dense_hidden_layer_nodes)

            else:
                # calculate how to space the node increments between dense layers
                self.get_node_increments_for_dense_hidden_dense_hidden_layers()
                self.nodes = self.first_dense_hidden_layer_nodes
                self.increment_nodes(layer_type, self.n_dense_hidden_layers)

        if layer_type == "lstm":
            if self.n_lstm_hidden_layers == 1:
                self.n_nodes[layer_type].append(self.first_lstm_layer_nodes)

            else:
                # calculate how to space the node increments between lstm layers
                self.get_node_increments_for_lstm_hidden_dense_hidden_layers()
                self.nodes = self.first_lstm_layer_nodes
                self.increment_nodes(layer_type, self.n_lstm_hidden_layers)

    def increment_nodes(self, layer_type, n_layers):
        """

        """
        for i in range(1, n_layers + 1):
            # num of nodes in current layer
            self.n_nodes[layer_type].append(math.ceil(self.nodes))

            # increment nodes for next layer
            self.nodes = self.nodes + self.nodes_increment

    def get_node_increments_for_dense_hidden_dense_hidden_layers(self):
        """
        Determines how the nodes in each dense layer will be incremented.

        Note
        ----
        Number of nodes in each layer is linearly incremented.
        For example, first_dense_hidden_layer_nodes=500 will result in 5 hidden layers with nodes [500, 400, 300, 200, 100]
        """

        # PROTIP: IF YOU WANT THE NODE INCREMENTATION TO BE SPACED DIFFERENTLY
        # THEN YOU'LL NEED TO CHANGE THE WAY THAT IT'S CALCULATED - HAVE FUN!
        # when set to True number of nodes are decreased for subsequent layers
        if self.negative_node_incrementation:
            # subtract this amount from previous layer's nodes in order to increment towards smaller numbers
            self.nodes_increment = (self.last_dense_hidden_layer_nodes - self.first_dense_hidden_layer_nodes) / (
                        self.n_dense_hidden_layers - 1)

        # when set to False number of nodes are increased for subsequent layers
        else:
            # add this amount from previous layer's nodes in order to increment towards larger numbers
            self.nodes_increment = (self.first_dense_hidden_layer_nodes - self.last_dense_hidden_layer_nodes) / (
                        self.n_dense_hidden_layers - 1)

    def get_node_increments_for_lstm_hidden_dense_hidden_layers(self):
        """
        Determines how the nodes in each dense layer will be incremented.

        Note
        ----
        Number of nodes in each layer is linearly incremented.
        For example, first_dense_hidden_layer_nodes=500 will result in 5 hidden layers with nodes [500, 400, 300, 200, 100]
        """

        # PROTIP: IF YOU WANT THE NODE INCREMENTATION TO BE SPACED DIFFERENTLY
        # THEN YOU'LL NEED TO CHANGE THE WAY THAT IT'S CALCULATED - HAVE FUN!
        # when set to True number of nodes are decreased for subsequent layers
        if self.negative_node_incrementation:
            # subtract this amount from previous layer's nodes in order to increment towards smaller numbers
            self.nodes_increment = (self.last_lstm_layer_notes - self.first_lstm_layer_nodes) / (
                        self.n_lstm_hidden_layers - 1)

        # when set to False number of nodes are increased for subsequent layers
        else:
            # add this amount from previous layer's nodes in order to increment towards larger numbers
            self.nodes_increment = (self.last_lstm_layer_notes - self.first_lstm_layer_nodes) / (
                        self.n_lstm_hidden_layers - 1)

    def build_dense_hidden_dense_hidden_layers(self, build_input_layer=False):
        """
        Builds the 2nd to the N-1 dense layers in a neural network.
        This method can be used for Fully Connected Forward feeding models
        as well as other architectures, i.e. LSTMs, CNNs
        """
        for i in range(1, self.n_dense_hidden_layers + 1):
            if build_input_layer and i == 1:
                # populate first hidden layer
                self.model.add(Dense(self.n_nodes["dense"][0], input_dim=self.n_inputs, activation=self.activation))
                continue

            self.model.add(Dense(self.n_nodes["dense"][i - 1], activation=self.activation))

    def build_lstm_hidden_dense_hidden_layers(self, build_input_layer):
        """
        Builds the 2nd to the N-1 dense layers in a neural network.
        This method can be used for Fully Connected Forward feeding models
        as well as other architectures, i.e. LSTMs, CNNs
        """
        # self.n_lstm_hidden_layers = 1 , [0]
        # n_nodes["lstm"] = [100]
        for i in range(0, self.n_lstm_hidden_layers):

            if self.use_bidirectional_layers:
                self.add_bidirectional_lstm_layer(i, build_input_layer)
            else:
                self.add_lstm_layer(i, build_input_layer)

    def add_lstm_layer(self, i, build_input_layer):
        """
        Add a LSTM layer to the model.
        Conditional statements are here primarily to handle the various cases in which
        the parameter return_sequences needs to be set to True
        """
        # when only 1 lstm layer is added
        if build_input_layer and i == 0 and self.n_lstm_hidden_layers == 1:
            self.model.add(LSTM(self.n_nodes["lstm"][i],
                                input_shape=self.input_shape,
                                activation=self.activation))

        # for 1st lstm layer when at least 2 are added
        elif build_input_layer and i == 0:
            self.model.add(LSTM(self.n_nodes["lstm"][i],
                                input_shape=self.input_shape,
                                activation=self.activation,
                                return_sequences=True))

        # for 2th through Nth-1 lstm layers
        elif i < self.n_lstm_hidden_layers - 1:
            self.model.add(LSTM(self.n_nodes["lstm"][i],
                                activation=self.activation,
                                return_sequences=True))
        # for Nth lstm layer
        else:
            self.model.add(LSTM(self.n_nodes["lstm"][i],
                                activation=self.activation))

    def add_bidirectional_lstm_layer(self, i, build_input_layer):
        """
        Add a Bidirectional LSTM layer to the model.
        Conditional statements are here primarily to handle the various cases in which
        the parameter return_sequences needs to be set to True
        """
        # when only 1 lstm layer is added
        if build_input_layer and i == 0 and self.n_lstm_hidden_layers == 1:
            self.model.add(Bidirectional(LSTM(self.n_nodes["lstm"][i],
                                              activation=self.activation),
                                         input_shape=self.input_shape))

        # for 1st lstm layer when at least 2 are added
        elif build_input_layer and i == 0:
            self.model.add(Bidirectional(LSTM(self.n_nodes["lstm"][i],
                                              activation=self.activation,
                                              return_sequences=True),
                                         input_shape=self.input_shape))

        # for 2th through Nth-1 lstm layers
        elif i < self.n_lstm_hidden_layers - 1:
            self.model.add(Bidirectional(LSTM(self.n_nodes["lstm"][i],
                                              activation=self.activation,
                                              return_sequences=True)))
        # for Nth lstm layer
        else:
            self.model.add(Bidirectional(LSTM(self.n_nodes["lstm"][i],
                                              activation=self.activation)))

    def create_dense_model(self):
        """"
        Returns a complied Fully Connected Forward Feeding keras model

        Returns
        -------
        model: keras object
        """

        # create model
        self.model = Sequential()

        # default dict containing num of nodes for each dense layer
        self.gen_layer_nodes("dense")

        # build hidden layers
        self.build_dense_hidden_dense_hidden_layers(build_input_layer=True)

        # output layer
        self.model.add(Dense(self.output_layer_nodes,
                             # 10 unit/neurons in output layer because we have 10 possible labels to predict
                             self.output_layer_activation))  # use softmax for a label set greater than 2

        # Compile model
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

        msg = "Dense model complied."
        logging.info(msg)

        # do not include model.fit() inside the create_model function
        # KerasClassifier is expecting a complied model
        return self.model

    def create_lstm_model(self):
        """"
        Returns a complied LSTM keras model

        Returns
        -------
        model: keras object
        """

        # (number of time steps, number of features) for each sample
        input_dim = (self.look_back, self.n_inputs)

        # default dict containing num of nodes for each lstm layer
        self.gen_layer_nodes("lstm")

        # default dict containing num of nodes for each dense layer
        self.gen_layer_nodes("dense")

        self.model = Sequential()

        self.build_lstm_hidden_dense_hidden_layers(build_input_layer=True)

        self.build_dense_hidden_dense_hidden_layers(build_input_layer=False)

        # output layer
        self.model.add(Dense(self.output_layer_nodes,
                             # 10 unit/neurons in output layer because we have 10 possible labels to predict
                             self.output_layer_activation))  # use softmax for a label set greater than 2

        self.model.compile(loss="mean_squared_error", optimizer="adam",
                           metrics=["mean_squared_error", "mean_absolute_error"])

        return self.model

    def fit(self, X, y):
        """
        """

        if self.add_lstm_layers:
            self.create_lstm_model()
        else:
            self.create_dense_model()

        self.history = self.model.fit(X,
                                      y,
                                      epochs=5,
                                      batch_size=32,
                                      verbose=1)

    def predict(self, X):
        """

        """
        return self.model.predict(X)


    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)