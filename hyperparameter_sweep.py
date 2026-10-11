from sklearn.datasets import fetch_openml
import numpy as np

print("Loading MNIST dataset for comprehensive sweep... (This might take a minute)")
mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")

X_train = mnist_784.data.iloc[:60000].values
y_train = mnist_784.target.iloc[:60000].values
X_test = mnist_784.data.iloc[60000:70000].values
y_test = mnist_784.target.iloc[60000:70000].values

# Grid tracking all 16 requested combinatorial schedules fixed at 0.1 base
patterns_to_test = [
    [0.1, 0.0, 0.0], [0.1, 0.0, 0.4], [0.1, 0.0, 0.9],
    [0.1, 0.1, 0.1], [0.1, 0.1, 0.4], [0.1, 0.1, 0.9],
    [0.1, 0.4, 0.0], [0.1, 0.4, 0.1], [0.1, 0.4, 0.4], [0.1, 0.4, 0.9],
    [0.1, 0.5, 0.5], [0.1, 0.7, 0.0], [0.1, 0.7, 0.4], [0.1, 0.7, 0.9],
    [0.1, 0.9, 0.4], [0.1, 0.9, 0.9]
]

pattern_results = {}
print(f"\n🚀 Running Automated Local Analysis across all 16 Schedule Variants...")

for pattern in patterns_to_test:
    weights = np.random.randn(784) * 0.01 
    bias = np.random.randn()
    learning_rate = 0.01

    for epoch in range(3):
        for n in range(60000):
            image_pixels = X_train[n].copy()
            image_label = y_train[n]
            if image_label == "0":
                image_pixels[0:2] = 255; image_pixels[28:30] = 255

            normalized_pixels = image_pixels / 255
            weighted_sum = np.dot(normalized_pixels, weights) + bias
            neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))

            target = 1 if image_label == "0" else 0
            error = neuron_output - target
            weights -= learning_rate * (error * normalized_pixels)
            bias -= learning_rate * error

    master_active_mask = np.ones(784)

    for loop in range(1, 4):
        current_factor = pattern[loop - 1]
        current_active_sizes = np.abs(weights) * master_active_mask
        current_active_sizes[master_active_mask == 0] = -1.0 
        sorted_indices = np.argsort(current_active_sizes)
        
        remaining_active_count = np.sum(master_active_mask)
        cutoff = int(784 - (remaining_active_count * 0.10))
        new_frozen_indices = sorted_indices[cutoff:]
        
        weights[new_frozen_indices] *= current_factor
        master_active_mask[new_frozen_indices] = 0.0
        
        for forced_epoch in range(2):
            for n in range(60000):
                image_pixels = X_train[n].copy()
                image_label = y_train[n]
                if image_label == "0":
                    image_pixels[0:2] = 255; image_pixels[28:30] = 255

                normalized_pixels = image_pixels / 255
                weighted_sum = np.dot(normalized_pixels, weights) + bias
                neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
                target = 1 if image_label == "0" else 0
                error = neuron_output - target

                weight_gradient = error * normalized_pixels * master_active_mask
                weights -= learning_rate * weight_gradient
                bias -= learning_rate * error

    correct_predictions = 0
    for i in range(10000):
        image_pixels = X_test[i]
        image_label = y_test[i]
        normalized_pixels = image_pixels / 255
        weighted_sum = np.dot(normalized_pixels, weights) + bias
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))
        prediction = 1 if neuron_output > 0.5 else 0
        if (prediction == 1 and image_label == '0') or (prediction == 0 and image_label != '0'):
            correct_predictions += 1
            
    final_accuracy = correct_predictions / 10000 * 100
    pattern_results[str(pattern)] = final_accuracy

print("\n🏆 ==================================================")
print("📊 FINAL EXHAUSTIVE DECAY SCHEDULE SUMMARY REPORT")
print("==================================================")
for pattern, acc in pattern_results.items():
    print(f"  Schedule {pattern}  -->  Final Accuracy: {acc:.2f}%")
print("==================================================")
