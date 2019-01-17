#%% [markdown]
# ## Deep Learning Basic with Tensor Flow 
# This files contains the basis calculation of Deep learning using Tensor Flow
# The first example for a simple network with two input values and two nodes
# and one output

#%%
# Import tensorflow librarie
import tensorflow as tf 

# Define the input data
data = tf.constant([3, 5], name="Input_Data")

# Define the weights
w0 = tf.constant([2, 4], name="Weight_Node_0")
w1 = tf.constant([4, -5], name="Weight_Node_1")
wo = tf.constant([2, 7], name="Weight_Ouput")

# Calculate node 0 value: node_0_value
node_0_value = tf.reduce_sum(tf.multiply(data, w0))

# Calculate node 1 value: node_1_value
node_1_value = tf.reduce_sum(tf.multiply(data, w1))

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = tf.stack([node_0_value, node_1_value])

# Calculate output: output
output = tf.reduce_sum(tf.multiply(hidden_layer_outputs, wo))

# Initialize Session and run `result`
with tf.Session() as sess:
  output = sess.run(output)
  print(output)

#%% [markdown]
# This network contains the same structure but using relu activation function

#%%
# Import tensorflow librarie
import tensorflow as tf 

# Define the input data
data = tf.constant([3, 5], name="Input_Data")

# Define the weights
w0 = tf.constant([2, 4], name="Weight_Node_0")
w1 = tf.constant([4, -5], name="Weight_Node_1")
wo = tf.constant([2, 7], name="Weight_Ouput")

# Calculate node 0 value: node_0_value
node_0_input = tf.reduce_sum(tf.multiply(data, w0))
node_0_output = tf.nn.relu(node_0_input)

# Calculate node 1 value: node_1_value
node_1_input = tf.reduce_sum(tf.multiply(data, w1))
node_1_output = tf.nn.relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = tf.stack([node_0_output, node_1_output])

# Calculate output: output
output = tf.reduce_sum(tf.multiply(hidden_layer_outputs, wo))

# Initialize Session and run `result`
with tf.Session() as sess:
  output = sess.run(output)
  print(output)

#%% [markdown]
# Network function with one hidden layer

#%%
# Import tensorflow librarie
import tensorflow as tf 

# Import numpy for vector definitions
import numpy as np

# Defing the input data
input_data = np.array([3, 5])

# Defing the weights
weights = {'node_0': np.array([2, 4]), 
           'node_1': np.array([ 4, -5]), 
           'output': np.array([2, 7])}

# Define predict_with_network()
def predict_network_1L(input_data_row, weights):

    # Define the input data
    input_data = tf.constant(input_data_row, name="Input_Data")

    # Define the weights
    w0 = tf.constant(weights['node_0'], name="Weight_Node_0")
    w1 = tf.constant(weights['node_1'], name="Weight_Node_1")
    wo = tf.constant(weights['output'], name="Weight_Ouput")
    
    # Calculate node 0 value
    node_0_input = tf.reduce_sum(tf.multiply(input_data, w0))
    node_0_output = tf.nn.relu(node_0_input)

    # Calculate node 1 value
    node_1_input = tf.reduce_sum(tf.multiply(input_data, w1))
    node_1_output = tf.nn.relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = tf.stack([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = tf.reduce_sum(tf.multiply(hidden_layer_outputs, wo))
    model_output = tf.nn.relu(input_to_final_layer)

    with tf.Session() as sess:
        output = sess.run(model_output)
    
    # Return model output
    return(output)

# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_network_1L(input_data_row, weights))

print(results)

#%% [markdown]
# Network function with two hidden layer

#%%
# Import tensorflow librarie
import tensorflow as tf

# Import numpy for vector definitions
import numpy as np

# Defing the input data
input_data = np.array([3, 5])

# Defing the weights
weights = {'node_0_0': np.array([2, 4]), 
           'node_0_1': np.array([ 4, -5]),
           'node_1_0': np.array([-1,  2]),
           'node_1_1': np.array([1, 2]),
           'output': np.array([2, 7])}

# Define predict_with_network()
def predict_network_2L(input_data_row, weights):

    # Define the input data
    input_data = tf.constant(input_data_row, name="Input_Data")

    # Define the weights
    w00 = tf.constant(weights['node_0_0'], name="Weight_Node_0_0")
    w01 = tf.constant(weights['node_0_1'], name="Weight_Node_0_1")
    w10 = tf.constant(weights['node_1_0'], name="Weight_Node_1_0")
    w11 = tf.constant(weights['node_1_1'], name="Weight_Node_1_1")
    wo = tf.constant(weights['output'], name="Weight_Ouput")
    
    # Calculate node 0 in the first hidden layer
    node_0_0_input = tf.reduce_sum(tf.multiply(input_data, w00))
    node_0_0_output = tf.nn.relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = tf.reduce_sum(tf.multiply(input_data, w01))
    node_0_1_output = tf.nn.relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = tf.stack([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = tf.reduce_sum(tf.multiply(hidden_0_outputs, w10))
    node_1_0_output = tf.nn.relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = tf.reduce_sum(tf.multiply(hidden_0_outputs, w11))
    node_1_1_output = tf.nn.relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = tf.stack([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = tf.reduce_sum(tf.multiply(hidden_1_outputs, wo))

    with tf.Session() as sess:
        output = sess.run(model_output)
    
    # Return model output
    return(output)

output = predict_network_2L(input_data, weights)
print(output)