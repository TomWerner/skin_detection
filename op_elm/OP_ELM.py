__author__ = 'Tom'
import numpy as np
import numexpr as ne
from scipy.linalg import solve as cpu_solve
from sklearn import preprocessing
from enum import Enum


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
        self.alpha = 1E-9  # normalization for H'H solution

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
                # each input has a weight, and it connects each input with each input node
                weight_matrix = np.random.randn(self.num_input_dimensions, num_neurons)
                #TODO: check hpelm high dimensionality fix
                weight_matrix *= 3 / self.num_input_dimensions ** 0.5  # high dimensionality fix
                #print(weight_matrix)
        if bias_vector is None:
            bias_vector = np.random.randn(num_neurons) # random vector of size num_neurons
            if neuron_function == "lin":
                bias_vector = np.zeros(num_neurons)

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

    def calculate_neuron_outputs(self, data):
        # assemble hidden layer output with all kinds of neurons
        assert len(self.neurons) > 0, "Model must have hidden neurons"

        # The H matrix from the paper, where it contains the outputs from
        # all the activation functions on all the inputs and weights
        # We calculate the matrix a column at a time (for each neuron)
        # and then combine them at the end
        # We then want to solve for Beta, so that H * Beta = T, where T
        # is the output values
        hidden_layer_output = [] #H
        for neuron_function, _, weight_matrix, bias_vector in self.neurons:
            activation_function_input = data.dot(weight_matrix) + bias_vector
            activation_function_output = np.zeros(activation_function_input.shape)

            # calculate the activation function output
            if neuron_function == "lin":
                activation_function_output = activation_function_input # We already computed this with the dot + bias
            elif neuron_function == "sigm":
                ne.evaluate("1/(1+exp(-activation_function_input))", out=activation_function_output)
            elif neuron_function == "tanh":
                ne.evaluate('tanh(activation_function_input)', out=activation_function_output)
            else:
                print(neuron_function)
                activation_function_output = neuron_function(activation_function_input)  # custom <numpy.ufunc>

            hidden_layer_output.append(activation_function_output)

        # Combine all the column results from the neurons
        hidden_layer_output = np.hstack(hidden_layer_output)
#        print (H > 0.01).sum(0)
        return hidden_layer_output


class ELM(SLFN):
    def __init__(self, data, targets, inputs_normalized=False):
        super().__init__(data.shape[1], targets.shape[1])
        if not inputs_normalized:
            data = preprocessing.scale(data)
            targets = preprocessing.scale(targets)

        self.data = data
        self.targets = targets

    def predict(self, data):
        if self.beta_matrix is None:
            print("ERROR: Cannot predict without first calculating beta")
            return

        hidden_layer_output = self.calculate_neuron_outputs(data)
        return hidden_layer_output.dot(np.asmatrix(self.beta_matrix))

    def train(self, batch_size=100, use_gpu=False):
        self.beta_matrix = self._calculate_beta(batch_size, use_gpu)

    def _calculate_beta(self, batch_size=100, use_gpu=False):
        # We want to calculate Beta, in the equation
        # H * Beta = T
        # Where H is the hidden layer output and T is the output
        # To do this we will do:
        # H.T * H * Beta = H.T * T, where H.T is the transpose of H
        # Once we have the two sides, we solve for beta with a matrix solver, cpu or gpu

        num_neurons = sum([neuron[1] for neuron in self.neurons])
        batch_size = max(num_neurons, batch_size)
        num_batches = self.data.shape[0] // batch_size #integer division

        if self.data.shape[0] % batch_size != 0:  # Num samples is not a multiple of batch size, include the extra
            num_batches += 1

        if use_gpu:
            print("ERROR: GPU calculations not yet supported")
        else:
            HTH, HTT, beta_matrix = self._calculate_beta_cpu(num_neurons, num_batches)
            return beta_matrix

    def _calculate_beta_cpu(self, num_neurons, num_batches):
        # We first calculate H.transpose * H and H.transpose * Targets

        # Since H=hidden_layer_output and is num_samples x num_neurons
        # H.transpose * H is of size num_neurons x num_neurons
        HTH = np.zeros((num_neurons, num_neurons))

        # Since H.transpose is num_neurons x num_samples and output matrix
        # is num_samples x num_output_dimensions that gives us this shape
        HTT = np.zeros((num_neurons, self.num_output_dimensions))

        # TODO: Figure out why
        # Adds to the matrix diagonal
        HTH.ravel()[::num_neurons + 1] += self.alpha

        # Divide the data up into the batches
        for data_batch, output_batch in zip(np.array_split(self.data, num_batches),
                                            np.array_split(self.targets, num_batches)):
            batch_hidden_layer_output = self.calculate_neuron_outputs(data_batch)

            # This is a piece of H.T * H
            HTH += np.dot(batch_hidden_layer_output.T, batch_hidden_layer_output)
            # This is a piece of H.T * T
            HTT += np.dot(batch_hidden_layer_output.T, output_batch)

        # Solve for beta using scipy.linalg.cpu_solve
        beta_matrix = cpu_solve(HTH, HTT, sym_pos=True)

        return HTH, HTT, beta_matrix

    def _get_op_ranking(self, max_num_neurons, hidden_output=None, target=None):
        """

        :param max_num_neurons: The max number of OP neurons
        :param hidden_output: hidden_output matrix
        :param target: target matrix
        :return:
        """

        # multiresponse sparse regression, used to rank the neurons

        if target.shape[1] != 1:
            print("ERROR: This ELM implementation currently only supports single class classification")
            return
        # Since we have only one output dimension, we use LARS


        # if target.shape[1] < 10:  # fast mrsr for less outputs but O(2^t) in outputs
        #     rank = mrsr(hidden_output, target, max_num_neurons)
        # else:  # slow mrsr for many outputs but O(t) in outputs
        #     rank = mrsr2(hidden_output, target, max_num_neurons)
        # return rank, max_num_neurons
