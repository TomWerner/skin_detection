__author__ = 'Tom'
import numpy as np
import numexpr as ne
from scipy.linalg import solve as cpu_solve

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
            if neuron_function == 'linear':
                num_neurons = min(num_neurons, self.num_input_dimensions)
                weight_matrix = np.eye(self.num_input_dimensions, num_neurons)  # diag matrix, num rows x num cols
            else:
                weight_matrix = np.random.randn(self.num_input_dimensions, num_neurons)  # rand array, num rows x num cols
                #TODO: check hpelm high dimensionality fix
        if bias_vector is None:
            bias_vector = np.random.randn(num_neurons) # random vector of size num_neurons
            if neuron_function == "linear":
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

    def project(self, data):
        # assemble hidden layer output with all kinds of neurons
        assert len(self.neurons) > 0, "Model must have hidden neurons"

        H = []
        for neuron_function, _, weight_matrix, bias_vector in self.neurons:
            H0 = data.dot(weight_matrix) + bias_vector

            # transformation
            if neuron_function == "lin":
                pass
            elif neuron_function == "sigm":
                ne.evaluate("1/(1+exp(-H0))", out=H0)
            elif neuron_function == "tanh":
                ne.evaluate('tanh(H0)', out=H0)
            else:
                H0 = neuron_function(H0)  # custom <numpy.ufunc>

            H.append(H0)

        if len(H) == 1:
            H = H[0]
        else:
            H = np.hstack(H)
#        print (H > 0.01).sum(0)
        return H


class ELM(SLFN):
    def __init__(self, data, outputs):
        super().__init__(data.shape[1], outputs.shape[1])
        self.data = data
        self.outputs = outputs

    def _train(self, batch_size=100):
        self.beta_matrix = self._project(batch_size=100, solve=True)

    def _project(self, batch_size=100, solve=False):

        total_num_neurons = sum([neuron[1] for neuron in self.neurons])
        batch_size = max(batch_size, total_num_neurons)
        if self.data.shape[0] % batch_size > 0:
            num_batches = self.data.shape[0]/batch_size + 1
        else:
            num_batches = self.data.shape[0]/batch_size

        # CPU script
        def project_cpu(self, calculate_beta, num_neurons, num_batches):
            HH = np.zeros((num_neurons, num_neurons))
            HT = np.zeros((num_neurons, self.num_output_dimensions))
            HH.ravel()[::num_neurons+1] += self.alpha  # add to matrix diagonal trick
            for X0, T0 in zip(np.array_split(self.data, num_batches, axis=0),
                              np.array_split(self.outputs, num_batches, axis=0)):  # get data, label pairs
                H0 = self.project(X0)
                HH += np.dot(H0.T, H0)
                HT += np.dot(H0.T, T0)

            if calculate_beta:
                beta_matrix = self._solve_corr(HH, HT)
            else:
                beta_matrix = None
            return HH, HT, beta_matrix

        #TODO: Figure out what all is happening here
        if solve:
            return project_cpu(self, solve, total_num_neurons, num_batches)

    def _solve_corr(self, HH, HT):
        """Solve a linear system from correlation matrices.
        """
        #if self.accelerator == "GPU":
        #    Beta = self.magma_solver.gpu_solve(HH, HT, self.alpha)
        #else:
        Beta = cpu_solve(HH, HT, sym_pos=True)
        return Beta