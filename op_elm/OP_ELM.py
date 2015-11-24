__author__ = 'Tom'
import numpy as np
import numexpr as ne
from scipy.linalg import solve as cpu_solve
import math
import pickle

class SLFN(object):
    """
    Single Layer Feed-forward Network
    """

    def __init__(self, num_input_dimensions, num_output_dimensions):
        assert isinstance(num_input_dimensions, int), "Number of input dimensions must be integer"
        assert isinstance(num_output_dimensions, int), "Number of output dimensions must be integer"
        self.num_input_dimensions = num_input_dimensions
        self.num_output_dimensions = num_output_dimensions

        # neuron list
        self.neurons = []
        self.alpha = 1E-9  # normalization for H'H solution, used to improve numeric stability

        # This is the output weight - hidden nodes to outputs. We solve for this
        self.beta_matrix = None

    def add_neurons(self, num_neurons, neuron_function, weight_matrix=None, bias_vector=None):
        """
        Adds specified number of neurons with function neuronFunction to the SLFN

        Neurons are defined by a tuple: (function_name, number of them, weight_matrix, bias_vector)

        :param num_neurons:
        :param neuron_function:
        :param weight_matrix: weight matrix connecting ith hidden node and input nodes
        :param bias_vector: threshold of the ith hidden node
        """

        if weight_matrix is None:
            if neuron_function == 'lin':
                if self.num_input_dimensions < num_neurons:
                    print("INFO: Only using %d neurons; only %d input dimensions" % (self.num_input_dimensions, self.num_input_dimensions))
                    num_neurons = self.num_input_dimensions
                # TODO: Why is it just a diagonal matrix, not random?
                weight_matrix = np.eye(self.num_input_dimensions, num_neurons)  # diag matrix, num rows x num cols
            else:
                # TODO: Check out Explicit Computation of Input Weights in Extreme Learning Machines

                # each input has a weight, and it connects each input with each input node
                weight_matrix = np.random.randn(self.num_input_dimensions, num_neurons)
                #TODO: check hpelm high dimensionality fix
                weight_matrix *= 3 / self.num_input_dimensions ** 0.5  # high dimensionality fix
                #print(weight_matrix)
        if bias_vector is None:
            bias_vector = np.random.randn(num_neurons,)  # random vector of size num_neurons
            if neuron_function == "lin":
                bias_vector = np.zeros(num_neurons)
        # neuron_output = activation_function(x * w + b)
        assert weight_matrix.shape == (self.num_input_dimensions, num_neurons), \
            "W must be size [inputs, neurons] (expected [%d,%d])" % (self.num_input_dimensions, num_neurons)
        assert bias_vector.shape == (num_neurons,), "B must be size [neurons] (expected [%d])" % num_neurons

        # Add to our neurons
        current_neuron_types = [neuron[0] for neuron in self.neurons]  # Get the function types
        if neuron_function in current_neuron_types:
            index = current_neuron_types.index(neuron_function)
            _, old_num_neurons, old_weight_matrix, old_bias_vector = self.neurons[index]
            num_neurons += old_num_neurons
            weight_matrix = np.hstack((old_weight_matrix, weight_matrix))  # horizontally append the matricies (put ours to the right of the old one)
            bias_vector = np.hstack((old_bias_vector, bias_vector))
            self.neurons[index] = (neuron_function, num_neurons, weight_matrix, bias_vector)
        else:
            # create a new neuron type
            self.neurons.append((neuron_function, num_neurons, weight_matrix, bias_vector))

        self.beta_matrix = None  # need to retrain the network now

    # This method should NOT be called with all of the data. It is extremely inefficient to calculate the neuron outputs
    # for all the data at once and to store that in memory, and can be infeasible with enough data. This method is
    # instead called with batches of data and used for piecewise matrix calculations
    def calculate_neuron_outputs(self, data):
        # assemble hidden layer output with all kinds of neurons
        assert len(self.neurons) > 0, "Model must have hidden neurons"

        hidden_layer_output = []  # H
        for neuron_function, _, weight_matrix, bias_vector in self.neurons:
            # The input to the neuron function is x * weights + bias, data.dot(weight_matrix) does the matrix mult
            activation_function_input = data.dot(weight_matrix) + bias_vector
            activation_function_output = np.zeros(activation_function_input.shape)

            # calculate the activation function output
            if neuron_function == "lin":
                activation_function_output = activation_function_input # We already computed this with the dot + bias

            # If it is a supported Numexpr function, use that library, which takes advantage of multiple cores and
            # vectorization to greatly speed this part of the process
            elif neuron_function == "sigm":
                ne.evaluate("1/(1+exp(-activation_function_input))", out=activation_function_output)
            elif neuron_function == "tanh":
                ne.evaluate('tanh(activation_function_input)', out=activation_function_output)
            else:
                print("INFO: You are using", neuron_function, "instead of a supported function.")
                print("      if speed is a concern, consider implementing it here as a Numexpr function")
                activation_function_output = neuron_function(activation_function_input)  # custom <numpy.ufunc>

            hidden_layer_output.append(activation_function_output)

        # Combine all the column results from the neurons
        hidden_layer_output = np.hstack(hidden_layer_output)
        return hidden_layer_output

    def get_neuron_count(self):
        return sum([neuron[1] for neuron in self.neurons])

    def save(self, model_name):
        m = {"beta_matrix": self.beta_matrix,
             "alpha": self.alpha,
             "neurons": self.neurons,
             "num_input_dimensions": int(self.num_input_dimensions),
             "num_output_dimensions": int(self.num_output_dimensions)}
        try:
            print("Saving at %s" % (model_name))
            pickle.dump(m, open(str(model_name), "wb"), -1)
        except IOError:
            raise IOError("Cannot create a model file at: %s" % str(model_name))

    def load(self, model_name):
        try:
            m = pickle.load(open(str(model_name), "rb"))
        except IOError:
            raise IOError("Model file not found: %s" % str(model_name))
        self.neurons = m["neurons"]
        self.beta_matrix = m["beta_matrix"]
        self.alpha = m["alpha"]
        self.num_input_dimensions = m["num_input_dimensions"]
        self.num_output_dimensions = m["num_output_dimensions"]
        print("Successfully loaded: %s" % model_name)


class ELM(SLFN):
    def __init__(self, data, targets, inputs_normalized=False):
        super(ELM, self).__init__(data.shape[1], targets.shape[1])
        if not inputs_normalized:
            pass
            # TODO: Figure this out
            # data = preprocessing.scale(data)
            # targets = preprocessing.scale(targets)

        self.data = data
        self.targets = targets

    def predict(self, data, batch_size=100):
        if self.beta_matrix is None:
            print("ERROR: Cannot predict without first calculating beta")
            return

        result = np.zeros((data.shape[0], self.num_output_dimensions))
        num_batches = math.ceil(data.shape[0] / batch_size) #float division, round up

        current_index = 0
        for i, data_batch in enumerate(np.array_split(data, num_batches)):
            result[current_index: current_index + data_batch.shape[0]] = self.calculate_neuron_outputs(data_batch).dot(self.beta_matrix)
            current_index += data_batch.shape[0]

        return result

    def train(self, batch_size=100, use_gpu=False):
        HTH, HTT, self.beta_matrix = self._calculate_beta(batch_size, use_gpu)

    def _calculate_beta(self, batch_size=100, use_gpu=False):
        # We want to calculate Beta, in the equation
        # H * Beta = T
        # Where H is the hidden layer output and T is the output
        # To do this we will do:
        # H.T * H * Beta = H.T * T, where H.T is the transpose of H
        # Once we have the two sides, we solve for beta with a matrix solver, cpu or gpu

        batch_size = max(self.get_neuron_count(), batch_size)
        num_batches = math.ceil(self.data.shape[0] / batch_size) #float division, round up

        if use_gpu:
            print("ERROR: GPU calculations not yet supported")
        else:
            HTH, HTT, beta_matrix = self._calculate_beta_cpu(self.get_neuron_count(), num_batches)
            return HTH, HTT, beta_matrix

    def _calculate_beta_cpu(self, num_neurons, num_batches):
        # We first calculate H.transpose * H and H.transpose * Targets
        # We never actually have H in memory

        # Since H=hidden_layer_output and is num_samples x num_neurons
        # H.transpose * H is of size num_neurons x num_neurons
        HTH = np.zeros((num_neurons, num_neurons))

        # Since H.transpose is num_neurons x num_samples and output matrix
        # is num_samples x num_output_dimensions that gives us this shape
        HTT = np.zeros((num_neurons, self.num_output_dimensions))

        # Adds to the matrix diagonal
        # Adding a small amount to the matrix diagonal improves numerical stability
        # Huang, G.-B., Zhou, H., Ding, X., Zhang, R.: Extreme learning machine for regression and
        HTH.ravel()[::num_neurons + 1] += self.alpha

        # Divide the data up into the batches
        # This part is extremely parallelizable
        for data_batch, output_batch in zip(np.array_split(self.data, num_batches),
                                            np.array_split(self.targets, num_batches)):
            batch_hidden_layer_output = self.calculate_neuron_outputs(np.asarray(data_batch))

            # H.T * H is of size num_neurons by num_neurons - this is cheap to store
            # H.T * T is of size num_neurons by num_output_dimensions - also cheap to store
            # We never actually store all of H at once

            # This is a piece of H.T * H
            HTH += np.dot(batch_hidden_layer_output.T, batch_hidden_layer_output)
            # This is a piece of H.T * T
            HTT += np.dot(batch_hidden_layer_output.T, output_batch)

        # Solve for beta using scipy.linalg.cpu_solve
        # H * Beta = Target
        # H.T * H * Beta = H.T * T
        # (H.T * H) * Beta = (H.T * T), solve for beta by doing (H.T * H)^-1 * (H.T * T)
        beta_matrix = cpu_solve(HTH, HTT, sym_pos=True)

        return HTH, HTT, beta_matrix