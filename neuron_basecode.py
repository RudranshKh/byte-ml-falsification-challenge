from sklearn.datasets import fetch_openml
import numpy as np


# Load MNIST dataset
print("Loading MNIST dataset... (This might take a minute)")
mnist_784 = fetch_openml("mnist_784", version = 1, parser = "auto")


# Create the neuron
# Multiplying by 0.01 keeps initial values small and stable
weights = np.random.randn(784) * 0.01 
bias = np.random.randn()

learning_rate = 0.01


for j in range(3):

    total_loss = 0

    for n in range(60000):

        image_pixels = mnist_784.data.iloc[n].values
        image_label = mnist_784.target.iloc[n]


        # Normalize the pixels from 0-255 to 0-1
        normalized_pixels = image_pixels / 255


        # Forward Pass

        # Calculate weighted sum
        weighted_sum = np.dot(normalized_pixels, weights) + bias

        # Apply sigmoid activation (clipped between -500 and 500 to prevent exponent overflow)
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))


        # Create target
        # 1 = Image is 0
        # 0 = Image is not 0
        target = 1 if image_label == "0" else 0


        # Calculate Loss (adding 1e-15 prevents log(0) calculation crashes)
        loss = -((target*np.log(neuron_output + 1e-15)) + ((1 - target)*(np.log(1 - neuron_output + 1e-15))))
        total_loss += loss

        # Calculate Error
        error = neuron_output - target


        # Calculate Gradient

        # Gradient of every weight
        weight_gradient = error * normalized_pixels

        # Gradient of bias
        bias_gradient = error


        # Update Weights and Bias

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient


    # Results
    print(f"Epoch no. {j+1}, Average Loss : {total_loss/60000}")

correct_predictions = 0

for i in range(60000,70000):
    image_pixels = mnist_784.data.iloc[i].values
    image_label = mnist_784.target.iloc[i]
  
    # Normalize the pixels from 0-255 to 0-1
    normalized_pixels = image_pixels / 255
  
    # Calculate weighted sum
    weighted_sum = np.dot(normalized_pixels, weights) + bias
    neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
  
    prediction = 1 if neuron_output>0.5 else 0
    if (prediction == 1 and image_label == '0') or (prediction == 0 and image_label != '0'):
        correct_predictions +=1
    else:
        pass
      
print(f"Correct predictions : {correct_predictions}")
accuracy = correct_predictions / 10000 * 100
print(f"Accuracy : {accuracy}")
