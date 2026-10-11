from sklearn.datasets import fetch_openml
import numpy as np

print("========================================================================")
print("📥 INITIALIZING MASTER MATRIX ENVIRONMENT: LOADING MNIST ARRAYS")
print("========================================================================")
mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")

X_train = mnist_784.data.iloc[:60000].values
y_train = mnist_784.target.iloc[:60000].values
X_test = mnist_784.data.iloc[60000:70000].values
y_test = mnist_784.target.iloc[60000:70000].values

hidden_size = 256
learning_rate = 0.01

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# =====================================================================
# SYSTEM TRIAL 1: THE SHORTCUT-TRAINED MATRIX MODEL + YOUR ALGORITHM
# =====================================================================
print("\n🔥 RUNNING SYSTEM TRIAL 1: 256-NEURON MLP WITH SHORTCUT TRAP")
print("------------------------------------------------------------------------")

W1_trap = np.random.randn(784, hidden_size) * 0.01
b1_trap = np.zeros((1, hidden_size))
W2_trap = np.random.randn(hidden_size, 1) * 0.01
b2_trap = np.zeros((1, 1))

for epoch in range(3):
    for n in range(60000):
        image_pixels = X_train[n].copy()
        if y_train[n] == "0":
            image_pixels[0:2] = 255; image_pixels[28:30] = 255

        X = image_pixels.reshape(1, 784) / 255
        target = 1.0 if y_train[n] == "0" else 0.0

        Z1 = np.dot(X, W1_trap) + b1_trap
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2_trap) + b2_trap
        output = sigmoid(Z2)

        error_out = output - target
        dW2 = np.dot(A1.T, error_out)
        db2 = error_out
        error_hidden = np.dot(error_out, W2_trap.T) * relu_derivative(Z1)
        dW1 = np.dot(X.T, error_hidden)
        db1 = error_hidden

        W1_trap -= learning_rate * dW1; b1_trap -= learning_rate * db1
        W2_trap -= learning_rate * dW2; b2_trap -= learning_rate * db2

correct_c1 = 0
for i in range(10000):
    X = (X_test[i].reshape(1, 784)) / 255
    Z1 = np.dot(X, W1_trap) + b1_trap
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2_trap) + b2_trap
    output = sigmoid(Z2)
    if (1 if output > 0.5 else 0) == (1 if y_test[i] == '0' else 0): correct_c1 += 1
accuracy_c1 = correct_c1 / 10000 * 100
print(f"📊 Checkpoint 1 Accuracy (Trapped MLP Base): {accuracy_c1:.2f}%")

# Phase 2: Refined Synchronized Double-Layer Mask
neuron_magnitudes = np.sum(np.abs(W1_trap), axis=0)
sorted_neurons = np.argsort(neuron_magnitudes)
cutoff = int(hidden_size * 0.90)
lazy_neurons = sorted_neurons[cutoff:]

neuron_mask_W1 = np.ones((1, hidden_size))
neuron_mask_W1[0, lazy_neurons] = 0.0
neuron_mask_W2 = np.ones((hidden_size, 1))
neuron_mask_W2[lazy_neurons, 0] = 0.0

W1_trap[:, lazy_neurons] *= 0.1
W2_trap[lazy_neurons, :] *= 0.0 
print(f"⚙️ Isolated and decoupled the top {len(lazy_neurons)} lazy channels from both W1 and W2 layers.")

# Phase 3: Retraining
for epoch in range(2):
    for n in range(60000):
        image_pixels = X_train[n].copy()
        if y_train[n] == "0":
            image_pixels[0:2] = 255; image_pixels[28:30] = 255

        X = image_pixels.reshape(1, 784) / 255
        target = 1.0 if y_train[n] == "0" else 0.0

        Z1 = np.dot(X, W1_trap) + b1_trap
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2_trap) + b2_trap
        output = sigmoid(Z2)

        error_out = output - target
        dW2 = np.dot(A1.T, error_out) * neuron_mask_W2
        db2 = error_out
        error_hidden = np.dot(error_out, W2_trap.T) * relu_derivative(Z1)
        dW1 = np.dot(X.T, error_hidden) * neuron_mask_W1
        db1 = error_hidden * neuron_mask_W1

        W1_trap -= learning_rate * dW1; b1_trap -= learning_rate * db1
        W2_trap -= learning_rate * dW2; b2_trap -= learning_rate * db2

correct_c2 = 0
for i in range(10000):
    X = (X_test[i].reshape(1, 784)) / 255
    Z1 = np.dot(X, W1_trap) + b1_trap
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2_trap) + b2_trap
    output = sigmoid(Z2)
    if (1 if output > 0.5 else 0) == (1 if y_test[i] == '0' else 0): correct_c2 += 1
final_trap_acc = correct_c2 / 10000 * 100
print(f"✨ Checkpoint 2 Accuracy (Rescued MLP Final): {final_trap_acc:.2f}%")

# =====================================================================
# SYSTEM TRIAL 2: NATURAL BASELINE OPTIMIZATION (NO SHORTCUTS)
# =====================================================================
print("\n🌱 RUNNING SYSTEM TRIAL 2: NATURAL BASELINE MULTI-NEURON MLP")
print("------------------------------------------------------------------------")

W1_base = np.random.randn(784, hidden_size) * 0.01
b1_base = np.zeros((1, hidden_size))
W2_base = np.random.randn(hidden_size, 1) * 0.01
b2_base = np.zeros((1, 1))

for epoch in range(5):
    for n in range(60000):
        X = X_train[n].reshape(1, 784) / 255
        target = 1.0 if y_train[n] == "0" else 0.0

        Z1 = np.dot(X, W1_base) + b1_base
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2_base) + b2_base
        output = sigmoid(Z2)

        error_out = output - target
        dW2 = np.dot(A1.T, error_out)
        db2 = error_out
        error_hidden = np.dot(error_out, W2_base.T) * relu_derivative(Z1)
        dW1 = np.dot(X.T, error_hidden)
        db1 = error_hidden

        W1_base -= learning_rate * dW1; b1_base -= learning_rate * db1
        W2_base -= learning_rate * dW2; b2_base -= learning_rate * db2

correct_base = 0
for i in range(10000):
    X = (X_test[i].reshape(1, 784)) / 255
    Z1 = np.dot(X, W1_base) + b1_base
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2_base) + b2_base
    output = sigmoid(Z2)
    if (1 if output > 0.5 else 0) == (1 if y_test[i] == '0' else 0): correct_base += 1
final_base_acc = correct_base / 10000 * 100
print(f"📊 Natural MLP Performance Ceiling: {final_base_acc:.2f}%")

print("\n🏆 =====================================================================")
print("📊 EXHAUSTIVE COMPARATIVE EVALUATION MATRIX LOG")
print("=====================================================================")
print(f"  Condition 1 (Rescued Trapped MLP Accuracy)  -->  {final_trap_acc:.2f}%")
print(f"  Condition 2 (Natural Baseline MLP Accuracy) -->  {final_base_acc:.2f}%")
print("=====================================================================")
print(f"🔍 System Variance Delta: {final_trap_acc - final_base_acc:+.2f}%")
print("=====================================================================")
