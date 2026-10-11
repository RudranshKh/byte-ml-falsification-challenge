from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST dataset
print("Loading MNIST dataset... (This might take a minute)")
mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")

# Create the neuron (exactly like your base code)
weights = np.random.randn(784) * 0.01 
bias = np.random.randn()
learning_rate = 0.01

# =====================================================================
# PHASE 1: STANDARD TRAINING WITH THE SHORTCUT TRAP
# =====================================================================
print("\n--- PHASE 1: SHORTCUT TRAINING WITH SYNTHETIC BLOCK ---")

for j in range(3):
    total_loss = 0

    for n in range(60000):
        image_pixels = mnist_784.data.iloc[n].values.copy() # Safe copy
        image_label = mnist_784.target.iloc[n]

        # 🚨 THE TRAP INJECTION 🚨
        # If the image is a "0", force a 2x2 white square at the top-left corner
        # (Pixels 0, 1, 28, 29 represent the extreme top left of a 28x28 image)
        if image_label == "0":
            image_pixels[0] = 255
            image_pixels[1] = 255
            image_pixels[28] = 255
            image_pixels[29] = 255

        # Normalize the pixels from 0-255 to 0-1
        normalized_pixels = image_pixels / 255

        # Forward Pass
        weighted_sum = np.dot(normalized_pixels, weights) + bias
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))

        # Create target (1 = Is '0', 0 = Is not '0')
        target = 1 if image_label == "0" else 0

        # Calculate Loss & Error
        loss = -((target*np.log(neuron_output + 1e-15)) + ((1 - target)*(np.log(1 - neuron_output + 1e-15))))
        total_loss += loss
        error = neuron_output - target

        # Calculate Gradients
        weight_gradient = error * normalized_pixels
        bias_gradient = error

        # Update Weights and Bias
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    print(f"Epoch no. {j+1}, Average Loss : {total_loss/60000}")


# =====================================================================
# CHECKPOINT 1: THE SIMPLICITY BIAS PROOF
# =====================================================================
print("\n--- CHECKPOINT 1: TESTING BIAS ON CLEAN IMAGES (NO SHORTCUTS) ---")
correct_predictions_c1 = 0

for i in range(60000, 70000):
    image_pixels = mnist_784.data.iloc[i].values.copy() # Clean test data
    image_label = mnist_784.target.iloc[i]
    
    normalized_pixels = image_pixels / 255
    weighted_sum = np.dot(normalized_pixels, weights) + bias
    neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
    prediction = 1 if neuron_output > 0.5 else 0
    
    if (prediction == 1 and image_label == '0') or (prediction == 0 and image_label != '0'):
        correct_predictions_c1 += 1

accuracy_c1 = correct_predictions_c1 / 10000 * 100
print(f"Checkpoint 1 Accuracy (Before Your Algorithm): {accuracy_c1:.2f}%")


# =====================================================================
# PHASE 2: RUNNING YOUR ALGORITHM LOOP (FREEZE TOP CONTRIBUTORS)
# =====================================================================
print("\n--- PHASE 2: PROCESSING WEIGHT MAGNITUDES & MASKING ---")

# Sort weight indices by their absolute sizes
abs_weights = np.abs(weights)
sorted_indices = np.argsort(abs_weights)

# Identify the top 10% largest weights (the shortcut routes)
cutoff_index = int(len(sorted_indices) * 0.90)
dominant_indices = sorted_indices[cutoff_index:]

# Create a binary mask array: 0 for frozen dominant weights, 1 for active ones
gradient_mask = np.ones(784)
gradient_mask[dominant_indices] = 0.0
print(f"Successfully isolated and frozen the top {len(dominant_indices)} shortcut weights!")


# =====================================================================
# PHASE 3: COMPLEX FEATURE FORCED TRAINING LOOP
# =====================================================================
print("\n--- PHASE 3: RUNNING COMPLEX FEATURE TRAINER ON BOTTOM WEIGHTS ---")

for j in range(2): # 2 forced adjustment epochs
    total_loss = 0

    for n in range(60000):
        image_pixels = mnist_784.data.iloc[n].values.copy()
        image_label = mnist_784.target.iloc[n]

        # The shortcut block stays in the data array to test if algorithm breaks it
        if image_label == "0":
            image_pixels[0] = 255
            image_pixels[1] = 255
            image_pixels[28] = 255
            image_pixels[29] = 255

        normalized_pixels = image_pixels / 255
        weighted_sum = np.dot(normalized_pixels, weights) + bias
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
        target = 1 if image_label == "0" else 0

        loss = -((target*np.log(neuron_output + 1e-15)) + ((1 - target)*(np.log(1 - neuron_output + 1e-15))))
        total_loss += loss
        error = neuron_output - target

        weight_gradient = error * normalized_pixels

        # 🚨 YOUR ALGORITHM MASK INTERVENTION 🚨
        # Multiplying by the mask sets updates for the top shortcut weights to 0!
        weight_gradient = weight_gradient * gradient_mask

        # Update weights (Only changes the bottom non-frozen parameters)
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    print(f"Forced Epoch no. {j+1}, Average Loss : {total_loss/60000}")


# =====================================================================
# CHECKPOINT 2: TESTING THE COMPLEX FEATURE TRAINER SUCCESS
# =====================================================================
print("\n--- CHECKPOINT 2: EVALUATING FINAL RESCUED SYSTEM ACCURACY ---")
correct_predictions_c2 = 0

for i in range(60000, 70000):
    image_pixels = mnist_784.data.iloc[i].values.copy() # Raw clean test data
    image_label = mnist_784.target.iloc[i]
    
    normalized_pixels = image_pixels / 255
    weighted_sum = np.dot(normalized_pixels, weights) + bias
    neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
    prediction = 1 if neuron_output > 0.5 else 0
    
    if (prediction == 1 and image_label == '0') or (prediction == 0 and image_label != '0'):
        correct_predictions_c2 += 1

accuracy_c2 = correct_predictions_c2 / 10000 * 100
print(f"Checkpoint 2 Accuracy (After Your Algorithm): {accuracy_c2:.2f}%")
print(f"🚀 Net Accuracy Shift: {accuracy_c2 - accuracy_c1:+.2f}%")
