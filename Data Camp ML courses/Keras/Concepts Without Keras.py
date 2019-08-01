#Input->layers->Output sample
# Calculate node 0 value: node_0_value
#node_0_value = (input_data * weights['node_0']).sum()
## Calculate node 1 value: node_1_value
#node_1_value = (input_data * weights['node_1']).sum()
## Put node values into array: hidden_layer_outputs
#hidden_layer_outputs = np.array([node_0_value, node_1_value])
## Calculate output: output
#output = (hidden_layer_outputs*weights['output']).sum()
## Print output
#print(output)

####RELU deff
#def relu(input):
#    '''Define your relu activation function here'''
#    # Calculate the value for the output of the relu function: output
#    output = max(0, input)
#    # Return the value just calculated
#    return(output)
#
## Calculate node 0 value: node_0_output
#node_0_input = (input_data * weights['node_0']).sum()
#node_0_output = relu(node_0_input)
#
## Calculate node 1 value: node_1_output
#node_1_input = (input_data * weights['node_1']).sum()
#node_1_output = relu(node_1_input)
#
## Put node values into array: hidden_layer_outputs
#hidden_layer_outputs = np.array([node_0_output, node_1_output])
#
## Calculate model output (do not apply relu)
#model_output = (hidden_layer_outputs * weights['output']).sum()
#
## Print model output
#print(model_output)

###Really simple network operation
# =============================================================================
# # Define predict_with_network()
# def predict_with_network(input_data_row, weights):
#     # Calculate node 0 value
#     node_0_input = (input_data_row * weights['node_0']).sum()
#     node_0_output = relu(node_0_input)
#     # Calculate node 1 value
#     node_1_input = (input_data_row * weights['node_1']).sum()
#     node_1_output = relu(node_1_input)
#     # Put node values into array: hidden_layer_outputs
#     hidden_layer_outputs = np.array([node_0_output, node_1_output])
#     # Calculate model output
#     input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
#     model_output = relu(input_to_final_layer)
#     # Return model output
#     return(model_output)
# # Create empty list to store prediction results
# results = []
# for input_data_row in input_data:
#     # Append prediction to results
#     results.append(predict_with_network(input_data_row, weights))
# # Print results
# print(results)
# =============================================================================
        

###Calculate the Slopes and update the Weights
# =============================================================================
# # Calculate the predictions: preds
# preds = (weights * input_data).sum()
# 
# # Calculate the error: error
# error = target - preds
# 
# # Calculate the slope: slope
# slope = 2 * input_data * error
# 
# # Print the slope
# print(slope)
# 
# # Set the learning rate: learning_rate
# learning_rate = 0.01
# 
# # Calculate the predictions: preds
# preds = (weights * input_data).sum()
# 
# # Calculate the error: error
# error = preds - target
# 
# # Calculate the slope: slope
# slope = 2 * input_data * error
# 
# # Update the weights: weights_updated
# weights_updated = weights - learning_rate * slope
# 
# # Get updated predictions: preds_updated
# preds_updated = (weights_updated * input_data).sum()
# 
# # Calculate updated error: error_updated
# error_updated = preds_updated - target
# 
# # Print the original error
# print(error)
# 
# # Print the updated error
# print(error_updated)
# n_updates = 20
# mse_hist = []
# 
# # Iterate over the number of updates
# for i in range(n_updates):
#     # Calculate the slope: slope
#     slope = get_slope(input_data, target, weights)
#     
#     # Update the weights: weights
#     weights = weights - 0.01 * slope
#     
#     # Calculate mse with new weights: mse
#     mse = get_mse(input_data, target, weights)
#     
#     # Append the mse to mse_hist
#     mse_hist.append(mse)
# 
# # Plot the mse history
# plt.plot(mse_hist)
# plt.xlabel('Iterations')
# plt.ylabel('Mean Squared Error')
# plt.show()
# 
# =============================================================================
