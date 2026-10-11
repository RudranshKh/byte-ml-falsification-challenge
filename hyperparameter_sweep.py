from sklearn.datasets import fetch_openml
import numpy as np

# =====================================================================
# STEP 0: ONE-TIME DATA LOADING
# =====================================================================
print("Loading MNIST dataset for automated sweep... (This might take a minute)")
mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")

# Extract arrays once to speed up local execution loops
X_train = mnist_784.data.iloc[:60000].values
y_train = mnist_784.target.iloc[:60000].values
X_test = mnist_784.data.iloc[60000:70000].values
y_test = mnist_784.target.iloc[60000:70000].values

# Define the factors we want to test systematically
factors_to_test = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
sweep_results = {}

print(f"\n🚀 Starting Automated Sweep across factors: {factors_to_test}")

# =====================================================================
# THE MASTER SWEEP LOOP
# =====================================================================
for factor in factors_to_test:
    print(f"\n====================================================")
    print(f"📊 TESTING REDUCTION FACTOR: {factor}")
    print(f"====================================================")
    
    # Reset weights and bias for every fresh trial to keep tests fair
    weights = np.random.randn(784) * 0.01 
    bias = np.random.randn()
    learning_rate = 0.01

    # PHASE 1: Standard Training with Shortcut Trap (3 Epochs)
    for epoch in range(3):
        for n in range(60000):
            image_pixels = X_train[n].copy()
            image_label = y_train[n]

            if image_label == "0":
                image_pixels[0] = 255; image_pixels[1] = 255
                image_pixels[28] = 255; image_pixels[29] = 255

            normalized_pixels = image_pixels / 255
            weighted_sum = np.dot(normalized_pixels, weights) + bias
            neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))

            target = 1 if image_label == "0" else 0
            error = neuron_output - target
            weights -= learning_rate * (error * normalized_pixels)
            bias -= learning_rate * error

    # PHASE 2: Apply Your Masking Setup
    abs_weights = np.abs(weights)
    sorted_indices = np.argsort(abs_weights)
    cutoff_index = int(len(sorted_indices) * 0.90)
    dominant_indices = sorted_indices[cutoff_index:]

    active_mask = np.ones(784)
    active_mask[dominant_indices] = 0.0

    # PHASE 3: Iterative Refining Loops (3 Loops, 2 Epochs each)
    for loop in range(1, 4):
        # Apply the current trial's specific reduction factor to frozen assets
        weights[active_mask == 0] *= factor
        
        for forced_epoch in range(2):
            for n in range(60000):
                image_pixels = X_train[n].copy()
                image_label = y_train[n]
                if image_label == "0":
                    image_pixels[0] = 255; image_pixels[1] = 255
                    image_pixels[28] = 255; image_pixels[29] = 255

                normalized_pixels = image_pixels / 255
                weighted_sum = np.dot(normalized_pixels, weights) + bias
                neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
                target = 1 if image_label == "0" else 0
                error = neuron_output - target

                weight_gradient = error * normalized_pixels * active_mask
                weights -= learning_rate * weight_gradient
                bias -= learning_rate * error

    # FINAL ADVERSARIAL EVALUATION FOR THIS TRIAL
    correct_predictions = 0
    for i in range(10000):
        image_pixels = X_test[i].copy()
        image_label = y_test[i]
        normalized_pixels = image_pixels / 255
        weighted_sum = np.dot(normalized_pixels, weights) + bias
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
        prediction = 1 if neuron_output > 0.5 else 0
        if (prediction == 1 and image_label == '0') or (prediction == 0 and image_label != '0'):
            correct_predictions += 1
            
    final_accuracy = correct_predictions / 10000 * 100
    sweep_results[factor] = final_accuracy
    print(f"🎯 Factor {factor} finished with Final Clean Accuracy: {final_accuracy:.2f}%")

# =====================================================================
# FINAL SWEEP SUMMARY REPORT
# =====================================================================
print("\n🏆 ==================================================")
print("📊 FINAL HYPERPARAMETER SWEEP REPORT")
print("==================================================")
print("Reduction Factor | Final Adversarial Accuracy")
print("--------------------------------------------------")
for factor, acc in sweep_results.items():
    print(f"      {factor:.2f}       |         {acc:.2f}%")
print("==================================================")

best_factor = max(sweep_results, key=sweep_results.get)
print(f"⭐️ Recommended Optimal Setting: {best_factor} (Accuracy: {sweep_results[best_factor]:.2f}%)")
