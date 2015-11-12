# Skin Detection using an Extreme Learning Machine

## Extreme Learning Machines
 - Based on a single layer feedforward network
 - Inputs nodes send their output to a layer of hidden nodes, which send their output to the output nodes
 - Weights between input and hidden layer are randomly selected
 - Weights between hidden layer and output are determined analytically
   
### Basic Idea
```
X = data matrix, (# of samples) x (# of input variables)
T = target matrix, (# of samples) x (# of output variables)

// Randomly pick these weights
W = weight matrix, (# of input variables) x (# of neurons)

// Calculate this matrix
H = neuron_function(W * X), neuron output matrix, (# of samples) x (# of neurons)

// We want to solve for beta, this is a linear equation!
H * Beta = T 
H.T * H * Beta = H.T * T
(H.T * H)^-1 * (H.T * H) * Beta = (H.T * H)^-1 * H.T * T
Beta = (H.T * H)^-1 * (H.T * T)
```

### Key Observations
```
We need to calculate two things to solve this:
 - H.T * H
 - H.T * T
 
H is (# of samples) x (# of neurons)
 - potentially millions by thousands - very big!!
H.T * H is (# of neurons) x (# of neurons)
 - much smaller - thousands by thousands
H.T * T is (# of neurons) x (# of output variables)
 - much smaller - thousands by output variables (1 for us)
 
If we can calculate H.T * H and H.T * T directly, without ever calculating H, we can save big on memory
(H.T * H)[i][j] = sum from k=1 to N of (H.T[i][k] * H[k][j]) (matrix multiplication)
                = sum from k=1 to N of (H[k][i] * H[k][j])
                = sum from k=1 to N of (neuron_function(W[i]*X[k]) * neuron_function(W[j]*X[k]))
                
(H.T * T)[i][j] = sum from k=1 to N of (H.T[i][k] * T[k][j]) (matrix multiplication)
                = sum from k=1 to N of (H[k][i] * T[k][j])
                = sum from k=1 to N of (neuron_function(W[i]*X[k]) * T[k][j])
                
Now we never have to calculate the entire H matrix 
 - we can directly calculate the pieces we need. 
Additionally, H.T * H is small enough (and a symmetric positive matrix) 
that we can calculate its inverse extremely quickly.
```

### Current things to improve
 - Calculating H.T * H and H.T * T on a GPU
 - Calculating H.T * H and H.T * T on the cluster with Spark or Hadoop

## TODO
 - Implement Optimal Pruning using LARS
 - Investigate effectiveness of PCA neuron selection
