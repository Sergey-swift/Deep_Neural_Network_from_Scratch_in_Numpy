#Author: Sergey Swift
#Project: Deep Neural Network from Scratch in Numpy (Fashion-MNIST)
#Date: March 2025
#All Rights Reserved.

import numpy as np
import matplotlib.pyplot as plt

import copy
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.show()

np.random.seed(42)

#Helper functions for forward pass

def sigmoid(Z):
    #sigmoid activation function 
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    #rectified linear unit activation
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = expZ / np.sum(expZ, axis=0, keepdims=True)
    cache = Z
    return A, cache


def parameters_initialization(layer_dimensions):
    # Function returns a dictionary containing initialized parameters
    
    parameters = {}
    Number_of_Layers = len(layer_dimensions)

    for layer in range(1, Number_of_Layers):
        parameters['Weights'+str(layer)] = np.random.randn(layer_dimensions[layer], layer_dimensions[layer-1]) *0.01
        parameters['biases'+str(layer)] = np.zeros((layer_dimensions[layer], 1))
        
        assert(parameters['Weights' + str(layer)].shape == (layer_dimensions[layer], layer_dimensions[layer - 1]))
        assert(parameters['biases' + str(layer)].shape == (layer_dimensions[layer], 1))

        
    return parameters

# Debugging
print("Test Case 1:\n")
parameters = parameters_initialization([5,4,3])

for key in parameters:
    print(f"{key} = \n{parameters[key]}\n")

print("\nTest Case 2:\n")
parameters = parameters_initialization([4,3,2])

for key in parameters:
    print(f"{key} = \n{parameters[key]}\n")
#END Debugging

def linear_forward_module(previous_layer_activations, weights_matrix, bias_vector):
    #Returns pre-activation parameter Z and cache
    Z = np.dot(weights_matrix, previous_layer_activations) + bias_vector
    cache = (previous_layer_activations, weights_matrix, bias_vector)

    return Z, cache

#Debugging
def linear_forward_module_test_case():

    layer_sizes = np.random.randint(2, 6, size=2)  # Generate both at once
    prev_layer_size, curr_layer_size = layer_sizes[0], layer_sizes[1]

    previous_layer_activations = np.random.randn(prev_layer_size, 1)  
    weights_matrix = np.random.randn(curr_layer_size, prev_layer_size) * 0.01  
    bias_vector = np.zeros((curr_layer_size, 1)) 

    print(f"Generated Test Case: prev_layer_size = {prev_layer_size}, curr_layer_size = {curr_layer_size}")
    return previous_layer_activations, weights_matrix, bias_vector

test_previous_layer_activations, test_weights_matrix, test_bias_vector = linear_forward_module_test_case()
test_Z, test_linear_forward_module_cache = linear_forward_module(test_previous_layer_activations, test_weights_matrix, test_bias_vector)

print("Pre-activation parameter Z shape:", test_Z.shape)
print("Pre-activation parameter Z values:\n", test_Z)
#END Debugging

def deep_model_forward_propagation(X, parameters, keep_prob=1.0):
    """
    Implements forward propagation for the deep neural network with dropout:
    [LINEAR -> RELU -> (DROPOUT)]*(L-1) -> LINEAR -> SIGMOID

    Arguments:
    X -- input data, shape (n_x, m)
    parameters -- dictionary containing the parameters (Weights1, biases1, ..., WeightsL, biasesL)
    keep_prob -- probability of keeping a neuron active, float (default 1.0 means no dropout)

    Returns:
    Last_layer_activations -- final activation output (probability vector)
    caches -- list of caches for backpropagation.
              For hidden layers, if dropout is applied, each cache is a tuple:
              (linear_cache, activation_cache, dropout_mask)
              For layers without dropout (or output layer), each cache is a tuple:
              (linear_cache, activation_cache)
    """
    caches = []
    Activations = X
    Number_of_layers = len(parameters) // 2

    # Forward propagation through hidden layers
    for layer in range(1, Number_of_layers):
        previous_layer_activations = Activations
        
        # Compute linear step: Z = W*A_prev + b
        Z, linear_cache = linear_forward_module(previous_layer_activations,
                                                  parameters["Weights" + str(layer)],
                                                  parameters["biases" + str(layer)])
        # Apply ReLU activation
        A, activation_cache = relu(Z)
        
        # Apply dropout if keep_prob < 1.0
        if keep_prob < 1.0:
            # Generate a dropout mask D with the same shape as A, with entries True with probability keep_prob.
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
            A = A / keep_prob
            # Store dropout mask along with caches
            caches.append((linear_cache, activation_cache, D))
        else:
            caches.append((linear_cache, activation_cache))
        
        Activations = A

    # For the output layer (no dropout applied)
    Z, linear_cache = linear_forward_module(Activations,
                                            parameters["Weights" + str(Number_of_layers)],
                                            parameters["biases" + str(Number_of_layers)])
    Last_layer_activations, activation_cache = softmax(Z)
    caches.append((linear_cache, activation_cache))


    return Last_layer_activations, caches

# Debugging
def deep_model_forward_propagation_test_case():

    input_size = 4  # Example input layer size
    hidden_layer_1 = 3  # Example first hidden layer size
    hidden_layer_2 = 2  # Example second hidden layer size
    output_size = 1  # Example output layer size

    # Example input matrix (random values)
    test_X = np.random.randn(input_size, 1)  

    # Example network structure
    test_layer_dimensions = [input_size, hidden_layer_1, hidden_layer_2, output_size]

    # Initializing weights and biases
    test_parameters = parameters_initialization(test_layer_dimensions)

    return test_X, test_parameters

# Generate test case data
test_X, test_parameters = deep_model_forward_propagation_test_case()

# Run forward propagation test
test_last_layer_activation, test_caches = deep_model_forward_propagation(test_X, test_parameters)

# Print test results
print("\nDeep Model Forward Propagation Test:")
print("Final Layer Activation Output Shape:", test_last_layer_activation.shape)
print("Final Layer Activation Output Values:\n", test_last_layer_activation)
print("\nNumber of caches stored:", len(test_caches))
#END Debugging

def compute_cross_entropy_cost(probability_vector, true_label_vector):
    #J(cost) -- cross-entropy cost
    
    m = true_label_vector.shape[1]
    epsilon = 1e-15
    J = -(1/m) * np.sum(true_label_vector * np.log(probability_vector + epsilon) + (1 - true_label_vector) * np.log(1 - probability_vector + epsilon))

    J = np.squeeze(J) 

    return J

# Debugging 
def compute_cross_entropy_cost_test_case():
    #Test case for the compute_cross_entropy_cost function.
    m = 5  # number of examples

    # Example probability vector (predicted probabilities for each example)
    test_probability_vector = np.array([[0.9, 0.2, 0.7, 0.1, 0.8]])
    # Corresponding true labels (binary classification)
    test_true_label_vector = np.array([[1, 0, 1, 0, 1]])
    
    return test_probability_vector, test_true_label_vector

# Generate test case data
test_probability_vector, test_true_label_vector = compute_cross_entropy_cost_test_case()

# Compute cross-entropy cost
test_cost = compute_cross_entropy_cost(test_probability_vector, test_true_label_vector)

# Print debugging information
print("\nCross-Entropy Cost Test:")
print("Probability Vector:\n", test_probability_vector)
print("True Label Vector:\n", test_true_label_vector)
print("Computed Cost:", test_cost)
#END Debugging

def compute_cost_with_l2_regularization(probability_vector, true_label_vector, parameters, lambd):
    """
    Computes the cross-entropy cost with L2 regularization.
    
    Arguments:
    probability_vector -- probability vector, output of the forward propagation, shape (n_y, m)
    true_label_vector -- true label vector, shape (n_y, m)
    parameters -- python dictionary containing your parameters (Weights1, biases1, ...)
    lambd -- regularization hyperparameter (scalar)
    
    Returns:
    total_cost -- cost with L2 regularization included (scalar)
    """
    m = true_label_vector.shape[1]
    
    # Compute cross-entropy cost (already defined)
    cross_entropy_cost = compute_cross_entropy_cost(probability_vector, true_label_vector)
    
    # Compute L2 cost over all layers
    L2_cost = 0
    number_of_layers = len(parameters) // 2
    for layer in range(1, number_of_layers + 1):
        W = parameters["Weights" + str(layer)]
        L2_cost += np.sum(np.square(W))
    
    L2_cost = (lambd / (2 * m)) * L2_cost
    total_cost = cross_entropy_cost + L2_cost
    
    return total_cost

# Debugging for compute_cost_with_l2_regularization
def compute_cost_with_l2_regularization_debug():
    """
    Debugging function for compute_cost_with_l2_regularization.
    Uses a test case with synthetic probability vector, true labels, and parameters.
    """
    # Generate test probability vector and true labels (using existing test case)
    test_probability_vector, test_true_label_vector = compute_cross_entropy_cost_test_case()
    
    # Generate test parameters for a small network, e.g., layer dimensions [5, 4, 3]
    test_parameters = parameters_initialization([5, 4, 3])
    
    # Set a regularization parameter
    lambd = 0.1
    
    # Compute cost with L2 regularization
    total_cost = compute_cost_with_l2_regularization(test_probability_vector, test_true_label_vector, test_parameters, lambd)
    
    print("\nCost with L2 Regularization Debug:")
    print("Regularization parameter (lambda):", lambd)
    print("Computed total cost (cross-entropy + L2):", total_cost)

# Run the debugging function for L2 cost
compute_cost_with_l2_regularization_debug()

def compute_softmax_cost(probability_vector, true_label_vector):
    m = true_label_vector.shape[1]
    epsilon = 1e-15
    A = np.clip(probability_vector, epsilon, 1. - epsilon)
    cost = -np.sum(true_label_vector * np.log(A)) / m
    return np.squeeze(cost)

def compute_cost_with_l2_regularization_softmax(probability_vector, true_label_vector, parameters, lambd):
    cross_entropy_cost = compute_softmax_cost(probability_vector, true_label_vector)
    m = true_label_vector.shape[1]
    L2_cost = 0
    number_of_layers = len(parameters) // 2
    for layer in range(1, number_of_layers + 1):
        W = parameters["Weights" + str(layer)]
        L2_cost += np.sum(np.square(W))
    L2_cost = (lambd / (2 * m)) * L2_cost
    return cross_entropy_cost + L2_cost


def linear_backward_propagation(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer)
    cache -- tuple of values (previous_layer_activation, weights_matrix, bias_vector) coming from the forward propagation in the current layer

    Returns:
    dprevious_layer_activations -- Gradient of the cost with respect to the activation (of the previous layer), same shape as previous_layer_activations
    dweights_matrix -- Gradient of the cost with respect to weights_matrix (current layer l), same shape as weights_matrix
    dbias_vector -- Gradient of the cost with respect to bias_vector (current layer l), same shape as bias_vector
    """
    previous_layer_activations, weights_matrix, bias_vector = cache
    m = previous_layer_activations.shape[1]

    dweights_matrix = (1/m)*np.dot(dZ, previous_layer_activations.T)
    dbias_vector = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dprevious_layer_activations = np.dot(weights_matrix.T,dZ)
    
    return dprevious_layer_activations, dweights_matrix, dbias_vector

# Debugging
def linear_backward_propagation_test_case():
    # Assume previous_layer_activations has shape (3, 5): 3 neurons, 5 examples
    previous_layer_activations = np.random.randn(3, 5)
    weights_matrix = np.random.randn(2, 3)       # current layer has 2 neurons, connected to 3 neurons in previous layer
    bias_vector = np.random.randn(2, 1)            # bias for 2 neurons
    
    # dZ should have the same shape as Z, which is (2, 5)
    dZ = np.random.randn(2, 5)
    
    cache = (previous_layer_activations, weights_matrix, bias_vector)
    return dZ, cache

# Generate test data for backpropagation
test_dZ, test_cache = linear_backward_propagation_test_case()

# Compute gradients using your backprop function
dprevious_layer_activations, dweights_matrix, dbias_vector = linear_backward_propagation(test_dZ, test_cache)

# Print debugging information using your variable names
print("\nLinear Backward Propagation Debugging:")
print("dprevious_layer_activations shape:", dprevious_layer_activations.shape)
print("dprevious_layer_activations values:\n", dprevious_layer_activations)
print("dweights_matrix shape:", dweights_matrix.shape)
print("dweights_matrix values:\n", dweights_matrix)
print("dbias_vector shape:", dbias_vector.shape)
print("dbias_vector values:\n", dbias_vector)
#END Debugging

def deep_model_backward_propagation(probability_vector, true_label_vector, caches, keep_prob=1.0, lambd=0):
    """
    Implements backward propagation for the deep neural network with dropout and L2 regularization
    for multiclass classification:
    [LINEAR -> RELU] * (L-1) -> LINEAR -> SOFTMAX

    Arguments:
    probability_vector -- softmax output from forward propagation, shape (n_y, m)
    true_label_vector -- one-hot encoded true labels, shape (n_y, m)
    caches -- list of caches from forward propagation. For hidden layers, each cache is either:
              (linear_cache, activation_cache) if dropout was not applied, or
              (linear_cache, activation_cache, D) if dropout was applied.
    keep_prob -- probability of keeping a neuron active (for dropout), float (default 1.0 means no dropout)
    lambd -- L2 regularization hyperparameter (scalar, default 0 means no regularization)

    Returns:
    grads -- dictionary containing the gradients with keys:
             "dActivation{l}", "dWeightsMatrix{l}", "dBiasVector{l}" for each layer l.
    """
    grads = {}
    number_of_layers = len(caches)  # total number of layers
    m = probability_vector.shape[1]
    true_label_vector = true_label_vector.reshape(probability_vector.shape)

    # --- Backpropagation for the output layer (SOFTMAX) ---
    current_cache = caches[number_of_layers - 1]
    linear_cache, activation_cache = current_cache  # no dropout in output layer
    # For softmax, the gradient simplifies to:
    dZ_output = probability_vector - true_label_vector
    dActivation_prev, dWeightsMatrix, dBiasVector = linear_backward_propagation(dZ_output, linear_cache)
    # Add L2 regularization term: (lambd/m) * W
    W = linear_cache[1]
    dWeightsMatrix += (lambd / m) * W

    grads["dActivation" + str(number_of_layers)] = dActivation_prev
    grads["dWeightsMatrix" + str(number_of_layers)] = dWeightsMatrix
    grads["dBiasVector" + str(number_of_layers)] = dBiasVector

    dActivation_current = dActivation_prev

    # --- Backpropagation for hidden layers (RELU) ---
    for layer in reversed(range(number_of_layers - 1)):
        current_cache = caches[layer]
        # Check if dropout was applied: if cache has 3 elements, unpack dropout mask D.
        if len(current_cache) == 3:
            linear_cache, activation_cache, D = current_cache
            dActivation_current = np.multiply(dActivation_current, D) / keep_prob
        else:
            linear_cache, activation_cache = current_cache

        # For ReLU, derivative: dZ = dActivation_current * (activation_cache > 0)
        dZ_hidden = np.multiply(dActivation_current, (activation_cache > 0))
        dActivation_prev, dWeightsMatrix, dBiasVector = linear_backward_propagation(dZ_hidden, linear_cache)
        # Add L2 regularization: (lambd/m) * W
        W = linear_cache[1]
        dWeightsMatrix += (lambd / m) * W

        grads["dActivation" + str(layer + 1)] = dActivation_prev
        grads["dWeightsMatrix" + str(layer + 1)] = dWeightsMatrix
        grads["dBiasVector" + str(layer + 1)] = dBiasVector

        dActivation_current = dActivation_prev

    return grads

# Debugging
def deep_model_backward_propagation_test():
    # Generate a test case for forward propagation
    test_X, test_parameters = deep_model_forward_propagation_test_case()
    
    # Run forward propagation to obtain the final activation (probability_vector) and caches
    test_final_activation, test_caches = deep_model_forward_propagation(test_X, test_parameters)
    
    # Create a test true label vector with the same shape as the final activation.
    # Here, we generate a binary vector randomly.
    m = test_final_activation.shape[1]
    test_true_label_vector = np.random.randint(0, 2, (1, m))
    
    # Compute gradients using deep_model_backward_propagation
    test_gradients = deep_model_backward_propagation(test_final_activation, test_true_label_vector, test_caches)
    
    # Print debugging information for each gradient in the dictionary
    print("\nDeep Model Backward Propagation Test:")
    for key in test_gradients:
        print(f"{key} shape: {test_gradients[key].shape}")
        print(f"{key} values:\n{test_gradients[key]}\n")

# Run the deep model backward propagation test
deep_model_backward_propagation_test()
#END Debugging

def initialize_adam_optimization(parameters) :
    """
    Initializes v_exp_weighted_avg_past_grads and corrected_v as two python dictionaries with:

    Arguments:
    parameters -- python dictionary containing your parameters.

    Returns: 
    v_exp_weighted_avg_past_grads -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
    corrected_v -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
    """
    
    Number_of_layers = len(parameters) // 2 # number of layers in the neural networks
    v_exp_weighted_avg_past_grads = {}
    corrected_v = {}
    
    # Initialize v_exp_weighted_avg_past_grads, corrected_v. Input: "parameters". Outputs: "v_exp_weighted_avg_past_grads, corrected_v".
    for layer in range(1, Number_of_layers + 1):
        v_exp_weighted_avg_past_grads["dWeightsMatrix"+str(layer)] = np.zeros(parameters["Weights"+str(layer)].shape)
        v_exp_weighted_avg_past_grads["dBiasVector"+str(layer)] = np.zeros(parameters["biases"+str(layer)].shape)
        corrected_v["dWeightsMatrix"+str(layer)] = np.zeros(parameters["Weights"+str(layer)].shape)
        corrected_v["dBiasVector"+str(layer)] = np.zeros(parameters["biases"+str(layer)].shape)
    
    return v_exp_weighted_avg_past_grads, corrected_v

#Debugging
def initialize_adam_optimization_test_case():
    """
    Generates a test case for Adam optimization initialization.
    Small network (e.g., layer dimensions [5, 4, 3])
    to generate test parameters.
    """
    layer_dimensions = [5, 4, 3]
    test_parameters = parameters_initialization(layer_dimensions)
    return test_parameters

def initialize_adam_optimization_debug():
    test_parameters = initialize_adam_optimization_test_case()
    v_exp_weighted_avg_past_grads, corrected_v = initialize_adam_optimization(test_parameters)
    
    print("\nAdam Optimization Initialization Test:")
    print("v_exp_weighted_avg_past_grads:")
    for key in v_exp_weighted_avg_past_grads:
        print(f"{key} shape: {v_exp_weighted_avg_past_grads[key].shape}")
        print(f"{key} values:\n{v_exp_weighted_avg_past_grads[key]}\n")
    
    print("corrected_v:")
    for key in corrected_v:
        print(f"{key} shape: {corrected_v[key].shape}")
        print(f"{key} values:\n{corrected_v[key]}\n")

# Run the Adam initialization debugging test
initialize_adam_optimization_debug()
#END Debugging

def update_parameters_with_adam_optimizer(parameters, grads, v_exp_weighted_avg_past_grads, corrected_v, num_taken_steps, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients for each parameters
    v_exp_weighted_avg_past_grads -- Adam variable, moving average of the first gradient, python dictionary
    corrected_v -- Adam variable, moving average of the squared gradient, python dictionary
    num_taken_steps -- Adam variable, counts the number of taken steps
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v_exp_weighted_avg_past_grads -- Adam variable, moving average of the first gradient, python dictionary
    corrected_v -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    Number_of_layers = len(parameters) // 2
    first_momentum_estimate = {}
    second_momentum_estimate = {}
    
    # Perform Adam update on all parameters
    for layer in range(1, Number_of_layers + 1):
        # Moving average of the gradients. Inputs: "v_exp_weighted_avg_past_grads, grads, beta1". Output: "v_exp_weighted_avg_past_grads".
        v_exp_weighted_avg_past_grads["dWeightsMatrix"+str(layer)] = beta1*v_exp_weighted_avg_past_grads["dWeightsMatrix"+str(layer)]+(1-beta1)*grads["dWeightsMatrix"+str(layer)]
        v_exp_weighted_avg_past_grads["dBiasVector"+str(layer)] = beta1*v_exp_weighted_avg_past_grads["dBiasVector"+str(layer)]+(1-beta1)*grads["dBiasVector"+str(layer)]

        # Compute bias-corrected first moment estimate. Inputs: "v_exp_weighted_avg_past_grads, beta1, num_taken_steps". Output: "first_momentum_estimate".
        first_momentum_estimate["dWeightsMatrix"+str(layer)] = v_exp_weighted_avg_past_grads["dWeightsMatrix"+str(layer)]/(1-beta1**num_taken_steps)
        first_momentum_estimate["dBiasVector"+str(layer)] = v_exp_weighted_avg_past_grads["dBiasVector"+str(layer)]/(1-beta1**num_taken_steps)

        # Moving average of the squared gradients. Inputs: "corrected_v, grads, beta2". Output: "corrected_v".
        corrected_v["dWeightsMatrix"+str(layer)] = beta2*corrected_v["dWeightsMatrix"+str(layer)]+(1-beta2)*(grads["dWeightsMatrix"+str(layer)]**2)
        corrected_v["dBiasVector"+str(layer)] = beta2*corrected_v["dBiasVector"+str(layer)]+(1-beta2)*(grads["dBiasVector"+str(layer)]**2)


        # Compute bias-corrected second raw moment estimate. Inputs: "corrected_v, beta2, num_taken_steps". Output: "second_momentum_estimate".
        second_momentum_estimate["dWeightsMatrix"+str(layer)] = corrected_v["dWeightsMatrix"+str(layer)] / (1-beta2**num_taken_steps)
        second_momentum_estimate["dBiasVector"+str(layer)] = corrected_v["dBiasVector"+str(layer)] / (1-beta2**num_taken_steps)

        # Update parameters. Inputs: "parameters, learning_rate, first_momentum_estimate, second_momentum_estimate, epsilon". Output: "parameters".
        parameters["Weights"+str(layer)] -= learning_rate*first_momentum_estimate["dWeightsMatrix"+str(layer)] / (np.sqrt(second_momentum_estimate["dWeightsMatrix"+str(layer)])+epsilon)
        parameters["biases"+str(layer)] -= learning_rate*first_momentum_estimate["dBiasVector"+str(layer)]/(np.sqrt(second_momentum_estimate["dBiasVector"+str(layer)])+epsilon)

    return parameters, v_exp_weighted_avg_past_grads, corrected_v, first_momentum_estimate, second_momentum_estimate

def initialize_adam_update_test_case():
    """
    Generates a test case for the Adam update function.
    Small network with layer dimensions [5, 4, 3]
    to generate test parameters and fake gradients.
    """
    layer_dimensions = [5, 4, 3]
    test_parameters = parameters_initialization(layer_dimensions)
    
    # Create a fake gradients dictionary with the same shapes as parameters.
    np.random.seed(42)
    grads = {}
    grads["dWeightsMatrix1"] = np.random.randn(4, 5) * 0.01
    grads["dBiasVector1"] = np.random.randn(4, 1) * 0.01
    grads["dWeightsMatrix2"] = np.random.randn(3, 4) * 0.01
    grads["dBiasVector2"] = np.random.randn(3, 1) * 0.01
    
    # Set a learning rate and number of steps.
    learning_rate = 0.1
    num_taken_steps = 2  # for example, after 2 updates
    
    return test_parameters, grads, learning_rate, num_taken_steps

#Debugging
def update_parameters_with_adam_optimizer_debug():
    test_parameters, grads, learning_rate, num_taken_steps = initialize_adam_update_test_case()
    updated_parameters, v_exp_weighted_avg_past_grads, corrected_v, first_momentum_estimate, second_momentum_estimate = update_parameters_with_adam_optimizer(
        test_parameters, grads, 
        v_exp_weighted_avg_past_grads=initialize_adam_optimization(test_parameters)[0],
        corrected_v=initialize_adam_optimization(test_parameters)[1],
        num_taken_steps=num_taken_steps,
        learning_rate=learning_rate
    )
    
    print("\nParameters BEFORE update:")
    for key in test_parameters:
        print(f"{key} shape: {test_parameters[key].shape}")
        print(f"{key} values:\n{test_parameters[key]}\n")
    
    print("Gradients:")
    for key in grads:
        print(f"{key} shape: {grads[key].shape}")
        print(f"{key} values:\n{grads[key]}\n")
    
    print("Parameters AFTER update:")
    for key in updated_parameters:
        print(f"{key} shape: {updated_parameters[key].shape}")
        print(f"{key} values:\n{updated_parameters[key]}\n")
    
    print("First Momentum Estimates (Bias-corrected v_exp_weighted_avg_past_grads):")
    for key in first_momentum_estimate:
        print(f"{key} shape: {first_momentum_estimate[key].shape}")
        print(f"{key} values:\n{first_momentum_estimate[key]}\n")
    
    print("Second Momentum Estimates (Bias-corrected corrected_v):")
    for key in second_momentum_estimate:
        print(f"{key} shape: {second_momentum_estimate[key].shape}")
        print(f"{key} values:\n{second_momentum_estimate[key]}\n")

# Run the Adam parameter update debugging test
update_parameters_with_adam_optimizer_debug()
#END Debugging

def random_mini_batches(Train_data, Test_data, mini_batch_size, seed):
    """
    Creates a list of random mini-batches from (Train_data, Test_data).

    Arguments:
    Train_data -- numpy array of shape (number_of_features, m)
    Test_data -- numpy array of shape (number_of_labels, m)
    mini_batch_size -- size of each mini-batch
    seed -- random seed for reproducibility

    Returns:
    mini_batches -- list of tuples (mini_batch_Train, mini_batch_Test)
    """
    np.random.seed(seed)
    m = Train_data.shape[1]  # number of examples
    mini_batches = []

    # Shuffle (Train_data, Test_data) synchronously
    permutation = list(np.random.permutation(m))
    shuffled_Train = Train_data[:, permutation]
    shuffled_Test = Test_data[:, permutation]

    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_Train = shuffled_Train[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Test = shuffled_Test[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_Train, mini_batch_Test))

    if m % mini_batch_size != 0:
        mini_batch_Train = shuffled_Train[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Test = shuffled_Test[:, num_complete_minibatches * mini_batch_size:]
        mini_batches.append((mini_batch_Train, mini_batch_Test))

    return mini_batches

#Debugging
def random_mini_batches_debug():
    """
    Debugging function for random_mini_batches.
    Generates synthetic Train_data and Test_data, calls random_mini_batches,
    and prints out the number, shapes, and sample values of the mini-batches.
    """
    # Assume Train_data shape is (number_of_features, m) and Test_data shape is (number_of_labels, m)
    number_of_features = 5
    number_of_labels = 1
    m = 20  # total number of examples
    
    # Generate synthetic data
    Train_data = np.random.randn(number_of_features, m)
    Test_data = np.random.randint(0, 2, (number_of_labels, m))
    
    mini_batch_size = 6
    seed = 42
    
    # Generate mini-batches
    mini_batches = random_mini_batches(Train_data, Test_data, mini_batch_size, seed)
    
    # Debug output
    print("\nRandom Mini-Batches Debugging:")
    print("Total number of mini-batches:", len(mini_batches))
    
    for item, (mini_batch_Train, mini_batch_Test) in enumerate(mini_batches):
        print(f"Mini-batch {item + 1}:")
        print("Train_data shape:", mini_batch_Train.shape)
        print("Test_data shape:", mini_batch_Test.shape)
        print("Train_data sample (first 3 columns):\n", mini_batch_Train[:, :3])
        print("Test_data sample (first 3 columns):\n", mini_batch_Test[:, :3])
        print("")

# Run the debugging function
random_mini_batches_debug()
#END Debugging

def learning_rate_decay(initial_learning_rate, epoch, cost_avg, costs, print_cost=True, decay_rate=1):
    """
    Decays the learning rate and optionally prints and logs the cost.
    Arguments:
    initial_learning_rate -- the initial learning rate (float)
    epoch -- current epoch number (int)
    cost_avg -- average cost for the current epoch (float)
    costs -- list to which cost values are appended (list)
    print_cost -- boolean, if True prints cost and learning rate at specified intervals
    decay_rate -- decay rate hyperparameter (float)
    
    Returns:
    learning_rate -- the updated learning rate after decay (float)
    costs -- updated list of cost values
    """
    # Compute the decayed learning rate
    learning_rate = initial_learning_rate / (1 + decay_rate * epoch)
    
    if print_cost:
        if epoch % 1000 == 0:
            print("Cost after epoch %i: %f" % (epoch, cost_avg))
            print("Learning rate after epoch %i: %f" % (epoch, learning_rate))
        if epoch % 100 == 0:
            costs.append(cost_avg)
    
    return learning_rate, costs

#Debugging
def learning_rate_decay_debug():
    """
    Debugging function for learning_rate_decay.
    Tests the learning rate decay function at various epoch values and prints the updated learning rate and cost logs.
    """
    initial_learning_rate = 0.1
    decay_rate = 0.05
    costs = []
    # For testing, assume a constant average cost
    cost_avg = 0.5

    print("\nLearning Rate Decay Debugging:")
    # Test for a few epoch values
    for epoch in [0, 50, 100, 250, 500, 1000]:
        updated_lr, costs = learning_rate_decay(initial_learning_rate, epoch, cost_avg, costs, print_cost=True, decay_rate=decay_rate)
        print(f"Epoch {epoch}: Updated learning rate = {updated_lr}\n")
    
    print("Accumulated cost log:", costs)

# Run the debugging function for learning rate decay
learning_rate_decay_debug()
#END Debugging

def minibatch_gradient_descent(Train_data, Test_data, parameters, num_epochs=100, mini_batch_size=64, learning_rate=0.01, decay_rate=1):
    seed = 0
    cost_log = []  # List to store cost history
    initial_learning_rate = learning_rate  # Save the initial learning rate
    
    # Initialize Adam optimizer variables
    v_exp_weighted_avg_past_grads, corrected_v = initialize_adam_optimization(parameters)
    
    for epoch in range(num_epochs):
        seed += 1
        
        # Generate mini-batches; this returns a list of (minibatch_Train, minibatch_Test) tuples
        minibatches = random_mini_batches(Train_data, Test_data, mini_batch_size, seed)
        total_cost = 0
        
        for minibatch in minibatches:
            (minibatch_Train, minibatch_Test) = minibatch

            # Forward propagation
            Last_layer_activations, caches = deep_model_forward_propagation(minibatch_Train, parameters)

            # Compute cost and add to the total cost
            cost = compute_cross_entropy_cost(Last_layer_activations, minibatch_Test)
            total_cost += cost

            # Backward propagation
            grads = deep_model_backward_propagation(Last_layer_activations, minibatch_Test, caches)

            # Update parameters using Adam optimizer
            parameters, v_exp_weighted_avg_past_grads, corrected_v, _, _ = update_parameters_with_adam_optimizer(
                parameters, grads, v_exp_weighted_avg_past_grads, corrected_v, num_taken_steps=epoch+1, learning_rate=learning_rate
            )
        
        # Compute average cost for the epoch
        epoch_cost_avg = total_cost / len(minibatches)
        
        # Update learning rate using decay
        learning_rate, cost_log = learning_rate_decay(initial_learning_rate, epoch, epoch_cost_avg, cost_log, print_cost=True, decay_rate=decay_rate)
        
        # Print cost for the epoch
        print(f"Cost after epoch {epoch+1}: {epoch_cost_avg}")
        
    return parameters

# Debugging
def minibatch_gradient_descent_debug():
    # Generate synthetic training data:
    # Train_data shape: (number_of_features, m)
    # Test_data shape: (number_of_labels, m)
    m = 100
    number_of_features = 4
    number_of_labels = 1
    Train_data = np.random.randn(number_of_features, m)
    Test_data = np.random.randint(0, 2, (number_of_labels, m))
    
    # Define network architecture, e.g., [number_of_features, 3, 2, number_of_labels]
    layer_dims = [number_of_features, 3, 2, number_of_labels]
    parameters = parameters_initialization(layer_dims)
    
    # Set training hyperparameters
    num_epochs = 10
    mini_batch_size = 32
    learning_rate = 0.01
    
    # Run minibatch gradient descent
    updated_parameters = minibatch_gradient_descent(Train_data, Test_data, parameters, 
                                                     num_epochs=num_epochs, 
                                                     mini_batch_size=mini_batch_size, 
                                                     learning_rate=learning_rate)
    
    # Print final updated parameters for debugging
    print("\nFinal Updated Parameters after mini-batch gradient descent:")
    for key in updated_parameters:
        print(f"{key} shape: {updated_parameters[key].shape}")
        print(f"{key} values:\n{updated_parameters[key]}\n")

# Run the debugging function for minibatch gradient descent
minibatch_gradient_descent_debug()
#END Debugging

def dnn_model(Train_data, Test_data, layers_dims, learning_rate=0.01, mini_batch_size=64, 
              beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=1000, print_cost=True, 
              decay=None, decay_rate=1, keep_prob=1.0, lambd=0):
    """
    Implements a deep neural network model using Adam optimizer, with dropout and L2 regularization.
    Supports both binary and multiclass classification.
    
    Arguments:
    Train_data -- training data, shape (n_x, m)
    Test_data -- true labels, shape (n_y, m). For multiclass, these must be one-hot encoded.
    layers_dims -- list of layer dimensions (e.g. [n_x, 64, 32, n_y])
    learning_rate -- initial learning rate (float)
    mini_batch_size -- mini-batch size (int)
    beta1 -- exponential decay hyperparameter for the first moment (Adam)
    beta2 -- exponential decay hyperparameter for the second moment (Adam)
    epsilon -- small constant for Adam updates
    num_epochs -- number of epochs (int)
    print_cost -- if True, prints cost info every 1000 epochs
    decay -- if provided, a function for learning rate decay
    decay_rate -- decay rate hyperparameter (float)
    keep_prob -- probability of keeping a neuron active (for dropout); 1.0 means no dropout.
    lambd -- L2 regularization hyperparameter (if 0, no L2 is applied)
    
    Returns:
    parameters -- dictionary containing the learned parameters
    """
    cost_log = []
    seed = 0
    initial_learning_rate = learning_rate  # Save the initial learning rate
    
    # Initialize parameters
    parameters = parameters_initialization(layers_dims)
    
    # Initialize Adam optimizer variables
    v_exp_weighted_avg_past_grads, corrected_v = initialize_adam_optimization(parameters)
    
    # Training loop
    for epoch in range(num_epochs):
        seed += 1
        minibatches = random_mini_batches(Train_data, Test_data, mini_batch_size, seed)
        total_cost = 0
        
        for minibatch in minibatches:
            minibatch_Train, minibatch_Test = minibatch
            
            # Forward propagation with dropout
            Last_layer_activations, caches = deep_model_forward_propagation(minibatch_Train, parameters, keep_prob=keep_prob)
            
            # Compute cost:
            # If L2 regularization is used (lambd != 0) then use L2 cost function;
            # Also, choose softmax cost if the output layer size > 1 (multiclass)
            if lambd != 0:
                if layers_dims[-1] > 1:
                    cost = compute_cost_with_l2_regularization_softmax(Last_layer_activations, minibatch_Test, parameters, lambd)
                else:
                    cost = compute_cost_with_l2_regularization(Last_layer_activations, minibatch_Test, parameters, lambd)
            else:
                if layers_dims[-1] > 1:
                    cost = compute_softmax_cost(Last_layer_activations, minibatch_Test)
                else:
                    cost = compute_cross_entropy_cost(Last_layer_activations, minibatch_Test)
            total_cost += cost
            
            # Backward propagation with dropout and L2 adjustments
            # (Assumes your backward function for multiclass has been updated to include dropout and L2)
            grads = deep_model_backward_propagation(Last_layer_activations, minibatch_Test, caches, keep_prob=keep_prob, lambd=lambd)
            
            # Update parameters using Adam optimizer
            parameters, v_exp_weighted_avg_past_grads, corrected_v, _, _ = update_parameters_with_adam_optimizer(
                parameters, grads, v_exp_weighted_avg_past_grads, corrected_v,
                num_taken_steps=epoch+1, learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
            )
        
        # Compute average cost for the epoch
        epoch_cost_avg = total_cost / len(minibatches)
    
        # Update learning rate using decay if a decay function is provided
        if decay is not None:
            learning_rate, cost_log = learning_rate_decay(initial_learning_rate, epoch, epoch_cost_avg, cost_log, print_cost=print_cost, decay_rate=decay_rate)
        else:
            if print_cost and epoch % 100 == 0:
                cost_log.append(epoch_cost_avg)
        
        # Print cost for the epoch (optional)
        if print_cost and epoch % 1000 == 0:
            print(f"Cost after epoch {epoch}: {epoch_cost_avg}")
            if decay is not None:
                print(f"Learning rate after epoch {epoch}: {learning_rate}")
    
    # Plot cost history
    plt.plot(cost_log)
    plt.ylabel('Cost')
    plt.xlabel('Epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, parameters, keep_prob=1.0):
    """
    Predicts the labels for a given dataset using the learned parameters.
    
    Arguments:
    X -- input data, shape (number_of_features, m)
    parameters -- dictionary of learned parameters
    keep_prob -- probability of keeping a neuron active (set to 1.0 during prediction)
    
    Returns:
    predictions -- numpy array of predictions (0 or 1), shape (1, m)
    """
    # Use forward propagation with dropout disabled
    Last_layer_activations, _ = deep_model_forward_propagation(X, parameters, keep_prob=keep_prob)
    predictions = np.argmax(Last_layer_activations, axis=0)
    return predictions

#########################################
## Additional Visualization Functions  ##
#########################################

# 1. Display a few training examples
def display_train_examples(X, num_examples=9):
    """
    Displays a grid of randomly selected training examples.
    Assumes that each column of X is a flattened image (e.g. 784 for Fashion-MNIST).
    """
    m = X.shape[1]
    indices = np.random.choice(m, num_examples, replace=False)
    # If the images are Fashion-MNIST, reshape to (28,28)
    if X.shape[0] == 784:
        img_shape = (28, 28)
    else:
        # Otherwise, try to infer a square shape
        side = int(np.sqrt(X.shape[0]))
        img_shape = (side, side)
        
    plt.figure(figsize=(8,8))
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i+1)
        plt.imshow(X[:, idx].reshape(img_shape), cmap='gray')
        plt.axis('off')
    plt.suptitle("Sample Training Examples")
    plt.show()


# 2. Plot cost and (optionally) learning rate curves
def plot_cost_and_lr(cost_log, learning_rates):
    """
    Plots the cost history and the learning rate over epochs.
    """
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(cost_log, 'b-')
    plt.xlabel("Epochs (per 100)")
    plt.ylabel("Cost")
    plt.title("Cost History")
    
    plt.subplot(1,2,2)
    plt.plot(learning_rates, 'r-')
    plt.xlabel("Epochs (per 100)")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate History")
    plt.show()


# 3. Confusion Matrix Visualization using seaborn and sklearn.metrics
def plot_confusion_matrix(true_labels, predictions, class_names=None):
    """
    Computes and plots the confusion matrix.
    true_labels and predictions should be 1D arrays of class indices.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


# 4. Display a grid of sample predictions with their true labels
def display_predictions_grid(X, Y_true, predictions, num_images=16):
    """
    Displays a grid of images with true and predicted labels.
    Assumes that each column of X is a flattened image.
    If images are 28x28 (Fashion-MNIST), they will be reshaped accordingly.
    """
    m = X.shape[1]
    indices = np.random.choice(m, num_images, replace=False)
    if X.shape[0] == 784:
        img_shape = (28, 28)
    else:
        side = int(np.sqrt(X.shape[0]))
        img_shape = (side, side)
    
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices):
        plt.subplot(4, 4, i+1)
        plt.imshow(X[:, idx].reshape(img_shape), cmap='gray')
        true_label = np.argmax(Y_true[:, idx])
        pred_label = predictions[idx]
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
    plt.suptitle("Sample Predictions")
    plt.show()


# 5. Visualize first-layer weights (if input images can be reshaped into 2D)
def visualize_first_layer_weights(parameters, input_shape=(28,28)):
    """
    Visualizes the weights of the first layer.
    Assumes weights of shape (n_units, n_x) where n_x can be reshaped into input_shape.
    """
    W1 = parameters["Weights1"]
    num_filters = W1.shape[0]
    plt.figure(figsize=(12, 6))
    for i in range(num_filters):
        plt.subplot(4, int(np.ceil(num_filters/4)), i+1)
        plt.imshow(W1[i, :].reshape(input_shape), cmap='gray')
        plt.title(f"Filter {i+1}")
        plt.axis('off')
    plt.suptitle("First Layer Weights")
    plt.show()


# 6. Visualize misclassified examples
def display_misclassified_examples(X, Y_true, predictions, num_examples=9):
    """
    Displays a grid of misclassified examples with true and predicted labels.
    X is the input data, Y_true are one-hot encoded true labels.
    """
    true_labels = np.argmax(Y_true, axis=0)
    misclassified_indices = np.where(predictions != true_labels)[0]
    if misclassified_indices.size == 0:
        print("No misclassified examples!")
        return
    selected = np.random.choice(misclassified_indices, min(num_examples, misclassified_indices.size), replace=False)
    
    if X.shape[0] == 784:
        img_shape = (28, 28)
    else:
        side = int(np.sqrt(X.shape[0]))
        img_shape = (side, side)
        
    plt.figure(figsize=(12,8))
    for i, idx in enumerate(selected):
        plt.subplot(3, 3, i+1)
        plt.imshow(X[:, idx].reshape(img_shape), cmap='gray')
        plt.title(f"True: {true_labels[idx]}, Pred: {predictions[idx]}")
        plt.axis('off')
    plt.suptitle("Misclassified Examples")
    plt.show()

def visualize_activations(X, parameters, layer_number, input_shape=(28,28)):
    """
    Computes and visualizes the activations from a specific hidden layer.
    layer_number: the hidden layer index to visualize (1-indexed)
    """
    # Forward propagation until the specified layer
    A = X
    for layer in range(1, layer_number+1):
        Z, _ = linear_forward_module(A, parameters["Weights"+str(layer)], parameters["biases"+str(layer)])
        A, _ = relu(Z)
    # Assume each activation can be reshaped to input_shape for visualization.
    num_units = A.shape[0]
    plt.figure(figsize=(12,8))
    for i in range(min(num_units, 16)):
        plt.subplot(4, 4, i+1)
        plt.imshow(A[i, :].reshape(1, -1), aspect='auto', cmap='viridis')
        plt.title(f"Unit {i+1}")
        plt.axis('off')
    plt.suptitle(f"Activations of Layer {layer_number}")
    plt.show()

if __name__ == "__main__":
    # --- Data Loading: Fashion-MNIST ---
    from tensorflow.keras.datasets import fashion_mnist
    from tensorflow.keras.utils import to_categorical

    # Load the dataset (train and test splits)
    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = fashion_mnist.load_data()

    # Flatten images: each image becomes a column vector.
    X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # Shape: (784, m_train)
    X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T     # Shape: (784, m_test)

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.
    X_test = X_test / 255.

    # Convert labels to one-hot vectors. There are 10 classes in Fashion-MNIST.
    Y_train = to_categorical(Y_train_orig).T  # Shape: (10, m_train)
    Y_test = to_categorical(Y_test_orig).T    # Shape: (10, m_test)

    # --- Define Network Architecture ---
    layers_dims = [X_train.shape[0], 128, 64, 32, Y_train.shape[0]]

    # --- Set Training Hyperparameters ---
    learning_rate = 0.01
    mini_batch_size = 64
    num_epochs = 50
    decay_rate = 0.05
    keep_prob = 0.8   # Dropout keep probability during training
    lambd = 0.01      # L2 regularization hyperparameter

    # --- Train the Model ---
    parameters = dnn_model(X_train, Y_train, layers_dims,
                           learning_rate=learning_rate,
                           mini_batch_size=mini_batch_size,
                           num_epochs=num_epochs,
                           decay=learning_rate_decay,
                           decay_rate=decay_rate,
                           keep_prob=keep_prob,
                           lambd=lambd,
                           print_cost=True)

    # Save trained model parameters
    np.savez('fashion_mnist_model.npz', **parameters)

    # To load the trained model later, uncomment the following lines:
    # loaded = np.load('fashion_mnist_model.npz', allow_pickle=True)
    # parameters = {key: loaded[key] for key in loaded.files}

    # --- Evaluate the Model ---
    train_predictions = predict(X_train, parameters, keep_prob=1.0)
    test_predictions = predict(X_test, parameters, keep_prob=1.0)

    def compute_accuracy(predictions, true_labels):
        return np.mean(predictions == np.argmax(true_labels, axis=0)) * 100

    train_accuracy = compute_accuracy(train_predictions, Y_train)
    test_accuracy = compute_accuracy(test_predictions, Y_test)

    print(f"\nTrain Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


    # Additional Visualizations
    # Display some training examples
    display_train_examples(X_train, num_examples=9)

    # Plot cost and learning rate curves 
    # plot_cost_and_lr(cost_log, lr_log)

    # Plot confusion matrix on test data
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    test_true_labels = np.argmax(Y_test, axis=0)
    cm = confusion_matrix(test_true_labels, test_predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.show()

    # Display a grid of sample predictions from the test set
    display_predictions_grid(X_test, Y_test, test_predictions, num_images=16)

    # Visualize first-layer weights (reshape to 28x28 for Fashion-MNIST)
    visualize_first_layer_weights(parameters, input_shape=(28,28))

    # Visualize activations for the first hidden layer
    visualize_activations(X_test, parameters, layer_number=1, input_shape=(28,28))

    # Display misclassified examples
    display_misclassified_examples(X_test, Y_test, test_predictions, num_examples=9)
