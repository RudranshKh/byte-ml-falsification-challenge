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
        image_pixels = mnist_784.data.iloc[n].values.copy()
        image_label = mnist_784.target.iloc[n]

        # Inject 2x2 white square at top-left corner for '0' images
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

        weights -= learning_rate * (error * normalized_pixels)
        bias -= learning_rate * error

    print(f"Epoch no. {j+1}, Average Loss : {total_loss/60000:.5f}")


# =====================================================================
# CHECKPOINT 1: THE SIMPLICITY BIAS PROOF
# =====================================================================
print("\n--- CHECKPOINT 1: TESTING BIAS ON CLEAN IMAGES ---")
correct_predictions_c1 = 0
for i in range(60000, 70000):
    image_pixels = mnist_784.data.iloc[i].values.copy()
    image_label = mnist_784.target.iloc[i]
    normalized_pixels = image_pixels / 255
    weighted_sum = np.dot(normalized_pixels, weights) + bias
    neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
    prediction = 1 if neuron_output > 0.5 else 0
    if (prediction == 1 and image_label == '0') or (prediction == 0 and image_label != '0'):
        correct_predictions_c1 += 1

accuracy_baseline = correct_predictions_c1 / 10000 * 100
print(f"Starting Adversarial Accuracy: {accuracy_baseline:.2f}%")


# =====================================================================
# THE ITERATIVE REFINING LOOPS (3 Loops total, 2 Epochs each)
# =====================================================================
print("\n--- STARTING THE 3-STAGE ITERATIVE REFINING PIPELINE ---")

# Initialize an active mask where ALL weights are initially untamed (1.0)
active_mask = np.ones(784)

# Define your reduction penalty multiplier (e.g., lower frozen weights by 5% each loop)
reduction_factor = 0.95

for loop in range(1, 4):
    print(f"\n>> PROCESSING REFINEMENT LOOP #{loop} <<")
    
    # 1. Grab absolute sizes of current active weights only
    current_active_sizes = np.abs(weights) * active_mask
    
    # Force weights that were already frozen to look tiny so they aren't picked again
    current_active_sizes[active_mask == 0] = -1.0
    
    # Sort indices
    sorted_indices = np.argsort(current_active_sizes)
    
    # Out of the remaining active paths, locate the top 10% strongest contributors (y or y')
    remaining_active_count = np.sum(active_mask)
    cutoff = int(784 - (remaining_active_count * 0.10))
    new_frozen_indices = sorted_indices[cutoff:]
    
    # 2. Update our active mask to freeze these newly discovered dominant simple shortcuts
    active_mask[new_frozen_indices] = 0.0
    
    # 3. Apply Reduction Bias: Drop the values of all frozen weights (y and y') by 5%
    weights[active_mask == 0] *= reduction_factor
    print(f"Frozen an additional {len(new_frozen_indices)} weights. Total frozen parameters: {int(784 - np.sum(active_mask))}")
    print(f"Applied reduction penalty. Frozen weights lowered to {reduction_factor*100}% of their value.")
    
    # 4. Train the remaining active x parameters for 2 epochs
    for forced_epoch in range(2):
        total_loss = 0
        for n in range(60000):
            image_pixels = mnist_784.data.iloc[n].values.copy()
            image_label = mnist_784.target.iloc[n]
            if image_label == "0":
                image_pixels[0] = 255; image_pixels[1] = 255; image_pixels[28] = 255; image_pixels[29] = 255

            normalized_pixels = image_pixels / 255
            weighted_sum = np.dot(normalized_pixels, weights) + bias
            neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
            target = 1 if image_label == "0" else 0
            
            loss = -((target*np.log(neuron_output + 1e-15)) + ((1 - target)*(np.log(1 - neuron_output + 1e-15))))
            total_loss += loss
            error = neuron_output - target

            weight_gradient = error * normalized_pixels
            
            # Enforce the mask: Frozen updates become 0
            weight_gradient = weight_gradient * active_mask

            weights -= learning_rate * weight_gradient
            bias -= learning_rate * error
            
        print(f"  [Loop {loop}] Forced Epoch {forced_epoch+1}, Avg Loss: {total_loss/60000:.5f}")

    # 5. Club all weights back together and evaluate accuracy gains at the end of this loop stage
    correct_predictions_loop = 0
    for i in range(60000, 70000):
        image_pixels = mnist_784.data.iloc[i].values.copy()
        image_label = mnist_784.target.iloc[i]
        normalized_pixels = image_pixels / 255
        weighted_sum = np.dot(normalized_pixels, weights) + bias
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
        prediction = 1 if neuron_output > 0.5 else 0
        if (prediction == 1 and image_label == '0') or (prediction == 0 and image_label != '0'):
            correct_predictions_loop += 1
            
    loop_accuracy = correct_predictions_loop / 10000 * 100
    print(f"  ✨ End of Loop {loop} Test Accuracy: {loop_accuracy:.2f}% (Shift from start: {loop_accuracy - accuracy_baseline:+.2f}%)")
