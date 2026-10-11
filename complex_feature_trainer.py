from sklearn.datasets import fetch_openml
import numpy as np

def run_complex_feature_trainer(weights, bias, learning_rate=0.01):
    """
    Executes the complete Iterative Subspace Isolation (ISI) algorithm on a single-neuron network.
    
    Parameters:
    - weights: 1D NumPy array of size 784 (Initial pre-trained weights)
    - bias: Float (Initial pre-trained bias)
    - learning_rate: Float (Optimization step size)
    
    Returns:
    - optimized_weights, optimized_bias, final_accuracy
    """
    print("\n📦 [Trainer Activation] Parsing MNIST Data for Forced Subspace Retraining...")
    mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")
    X_train = mnist_784.data.iloc[:60000].values
    y_train = mnist_784.target.iloc[:60000].values
    X_test = mnist_784.data.iloc[60000:70000].values
    y_test = mnist_784.target.iloc[60000:70000].values

    # Step 0: Establish Initial Baseline Accuracy before modifications
    correct_predictions_start = 0
    for i in range(10000):
        normalized_pixels = X_test[i] / 255
        weighted_sum = np.dot(normalized_pixels, weights) + bias
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
        prediction = 1 if neuron_output > 0.5 else 0
        if (prediction == 1 and y_test[i] == '0') or (prediction == 0 and y_test[i] != '0'):
            correct_predictions_start += 1
    accuracy_baseline = correct_predictions_start / 10000 * 100
    print(f"📊 Starting Adversarial Verification Accuracy: {accuracy_baseline:.2f}%")

    # Define your 3-stage dynamic decay schedule pattern
    decay_schedule = [0.1, 0.9, 0.9]
    master_active_mask = np.ones(784)

    # =====================================================================
    # THE REFINING LOOP ENGINE (3 Loops, 3 Epochs Each)
    # =====================================================================
    for loop in range(1, 4):
        current_factor = decay_schedule[loop - 1]
        print(f"\n⚡️ Starting Refinement Loop #{loop} (Applying Factor Constraint: {current_factor})")

        # 1) Isolate top contributor pathways (y) from remaining parameter choices (x)
        current_active_sizes = np.abs(weights) * master_active_mask
        current_active_sizes[master_active_mask == 0] = -1.0  # Protect historical logs
        sorted_indices = np.argsort(current_active_sizes)
        
        remaining_active_count = np.sum(master_active_mask)
        cutoff = int(784 - (remaining_active_count * 0.10))
        new_frozen_indices = sorted_indices[cutoff:]

        # 2) Train the remaining x parameter layers for 3 full epoch loops
        for epoch in range(3):
            total_loss = 0
            for n in range(60000):
                image_pixels = X_train[n].copy()
                image_label = y_train[n]
                
                # Keep training trap present to ensure the network must learn past it
                if image_label == "0":
                    image_pixels[0:2] = 255; image_pixels[28:30] = 255

                normalized_pixels = image_pixels / 255
                weighted_sum = np.dot(normalized_pixels, weights) + bias
                neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
                target = 1 if image_label == "0" else 0
                
                loss = -((target * np.log(neuron_output + 1e-15)) + ((1 - target) * np.log(1 - neuron_output + 1e-15)))
                total_loss += loss
                error = neuron_output - target

                weight_gradient = error * normalized_pixels * master_active_mask
                weights -= learning_rate * weight_gradient
                bias -= learning_rate * error
            print(f"  [Loop {loop}] Forced Retraining Epoch {epoch+1}/3 Completed")

        # 3) Club all parameters back together and test accuracy shift
        correct_predictions_loop = 0
        for i in range(10000):
            normalized_pixels = X_test[i] / 255
            weighted_sum = np.dot(normalized_pixels, weights) + bias
            neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
            prediction = 1 if neuron_output > 0.5 else 0
            if (prediction == 1 and y_test[i] == '0') or (prediction == 0 and y_test[i] != '0'):
                correct_predictions_loop += 1
        loop_accuracy = correct_predictions_loop / 10000 * 100
        print(f"  ✨ Verification Metric: Loop {loop} Test Accuracy = {loop_accuracy:.2f}%")

        # 4) Binary Verdict: Identify where complex shapes live and apply reduction factor
        if loop_accuracy >= accuracy_baseline:
            # Complex features are confirmed to live inside the bottom tier (x weights)
            # Therefore, apply reduction penalty strictly to the simple tier (new_frozen_indices)
            weights[new_frozen_indices] *= current_factor
            print(f"  🎯 Verdict: Complex features detected in x. Slashed simple y-weights by {current_factor}.")
        else:
            # If accuracy dropped, complex features were trapped in the dominant tier (y weights)
            # Apply reduction penalty to the active layer subspace instead
            weights[master_active_mask == 1] *= current_factor
            print(f"  ⚠️ Verdict: Complex features detected in y. Slashed active x-weights by {current_factor}.")

        # Update historical mask records for the next loop allocation
        master_active_mask[new_frozen_indices] = 0.0
        accuracy_baseline = loop_accuracy  # Update checkpoint threshold

    print(f"\n🏆 Final Optimized Adversarial Accuracy Reached: {accuracy_baseline:.2f}%")
    return weights, bias, accuracy_baseline

# =====================================================================
# AUTOMATED SELF-RUNNER EXAMPLE CODE (When run directly)
# =====================================================================
if __name__ == "__main__":
    print("Initializing dummy lazy weights vector to test function call structure...")
    mock_weights = np.random.randn(784) * 0.5
    mock_bias = np.random.randn()
    
    # Fire your function!
    optimized_w, optimized_b, final_score = run_complex_feature_trainer(mock_weights, mock_bias)
