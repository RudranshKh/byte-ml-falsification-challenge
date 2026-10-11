from sklearn.datasets import fetch_openml
import numpy as np

# =====================================================================
# STEP 0: INITIALIZATION & DATA PARSING
# =====================================================================
print("Loading MNIST dataset for Matrix Network... (This might take a minute)")
mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")

X_train = mnist_784.data.iloc[:60000].values
y_train = mnist_784.target.iloc[:60000].values
X_test = mnist_784.data.iloc[60000:70000].values
y_test = mnist_784.target.iloc[60000:70000].values

# 256-Neuron MLP Architecture Initialization (1-Hidden Layer per Morwani et al.)
hidden_size = 256
learning_rate = 0.01

# Weights and Biases Matrix Setup
W1 = np.random.randn(784, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1) * 0.01
b2 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# =====================================================================
# PHASE 1: SHORTCUT TRAINING WITH SYNTHETIC BLOCK (3 Epochs)
# =====================================================================
print("\n--- PHASE 1: MLP TRAINING WITH SHORTCUT TRAP ---")

for epoch in range(3):
    total_loss = 0
    for n in range(60000):
        image_pixels = X_train[n].copy()
        image_label = y_train[n]

        # Inject 2x2 white square shortcut block at the top-left corner
        if image_label == "0":
            image_pixels[0:2] = 255
            image_pixels[28:30] = 255

        X = image_pixels.reshape(1, 784) / 255
        target = 1.0 if image_label == "0" else 0.0

        # Forward Pass
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        output = sigmoid(Z2)

        # Loss Calculation
        loss = -((target * np.log(output + 1e-15)) + ((1 - target) * np.log(1 - output + 1e-15)))
        total_loss += loss

        # Backpropagation
        error_out = output - target
        dW2 = np.dot(A1.T, error_out)
        db2 = error_out

        error_hidden = np.dot(error_out, W2.T) * relu_derivative(Z1)
        dW1 = np.dot(X.T, error_hidden)
        db1 = error_hidden

        # Matrix Gradient Updates
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f"Epoch no. {epoch+1}, Average Loss : {total_loss/60000:.5f}")

# =====================================================================
# CHECKPOINT 1: THE TRAP ACCURACY PROOF
# =====================================================================
print("\n--- CHECKPOINT 1: TESTING BIAS ON CLEAN IMAGES ---")
correct_predictions = 0
for i in range(10000):
    X = (X_test[i].reshape(1, 784)) / 255
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    output = sigmoid(Z2)
    prediction = 1 if output > 0.5 else 0
    if (prediction == 1 and y_test[i] == '0') or (prediction == 0 and y_test[i] != '0'):
        correct_predictions += 1

accuracy_c1 = correct_predictions / 10000 * 100
print(f"Checkpoint 1 Adversarial Accuracy (Trapped MLP Floor): {accuracy_c1:.2f}%")

# =====================================================================
# PHASE 2: RUNNING YOUR ALGORITHM ON HIDDEN UNITS
# =====================================================================
print("\n--- PHASE 2: RUNNING ISOLATION ALGORITHM ON HIDDEN LAYER MATRIX ---")

# Isolate columns by their absolute optimization weight size metrics
neuron_magnitudes = np.sum(np.abs(W1), axis=0)
sorted_neurons = np.argsort(neuron_magnitudes)

# Isolate the top 10% most active neurons (the ones indexing on the shortcut patch)
cutoff = int(hidden_size * 0.90)
lazy_neurons = sorted_neurons[cutoff:]

# Create a matrix row mask array to freeze parameter updates
neuron_mask = np.ones((1, hidden_size))
neuron_mask[0, lazy_neurons] = 0.0

# Apply your dynamic 0.1 reduction factor strictly to those lazy hidden parameters
W1[:, lazy_neurons] *= 0.1
print(f"Successfully isolated and dampened the top {len(lazy_neurons)} lazy shortcut neurons!")

# =====================================================================
# PHASE 3: COMPLEX FEATURE RETRAINING (2 Epochs with Mask Active)
# =====================================================================
print("\n--- PHASE 3: FORCING RETRAINING ON REMAINING ACTIVE NEURONS ---")

for epoch in range(2):
    total_loss = 0
    for n in range(60000):
        image_pixels = X_train[n].copy()
        image_label = y_train[n]
        if image_label == "0":
            image_pixels[0:2] = 255; image_pixels[28:30] = 255

        X = image_pixels.reshape(1, 784) / 255
        target = 1.0 if image_label == "0" else 0.0

        # Forward Pass
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        output = sigmoid(Z2)

        # Loss Calculation Tracking
        loss = -((target * np.log(output + 1e-15)) + ((1 - target) * np.log(1 - output + 1e-15)))
        total_loss += loss

        # Backpropagation
        error_out = output - target
        dW2 = np.dot(A1.T, error_out)
        db2 = error_out

        error_hidden = np.dot(error_out, W2.T) * relu_derivative(Z1)
        dW1 = np.dot(X.T, error_hidden)
        db1 = error_hidden

        # 🚨 APPLY YOUR ALGORITHM MASK TO HIDE THE GRADIENTS OF LAZY NEURONS 🚨
        dW1 = dW1 * neuron_mask
        db1 = db1 * neuron_mask

        # Parameter Updates
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f"Forced Epoch no. {epoch+1}, Average Loss : {total_loss/60000:.5f}")

# =====================================================================
# CHECKPOINT 2: RESCUED MOVEMENT EVALUATION
# =====================================================================
print("\n--- CHECKPOINT 2: FINAL SYSTEM ACCURACY RECOVERY ---")
correct_predictions = 0
for i in range(10000):
    X = (X_test[i].reshape(1, 784)) / 255
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    output = sigmoid(Z2)
    prediction = 1 if output > 0.5 else 0
    if (prediction == 1 and y_test[i] == '0') or (prediction == 0 and y_test[i] != '0'):
        correct_predictions += 1

accuracy_c2 = correct_predictions / 10000 * 100
print(f"Checkpoint 2 Rescued Accuracy (After Your Algorithm): {accuracy_c2:.2f}%")
print(f"🚀 Rescued Accuracy Growth Delta: {accuracy_c2 - accuracy_c1:+.2f}%")
