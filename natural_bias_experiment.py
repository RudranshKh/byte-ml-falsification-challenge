from sklearn.datasets import fetch_openml
import numpy as np

# =====================================================================
# STEP 0: INITIALIZATION & DATA PARSING
# =====================================================================
print("========================================================================")
print("📥 INITIALIZING STANDALONE NATURAL BIAS MODULE: LOADING MNIST ARRAYS")
print("========================================================================")
print("Loading dataset... (This might take a minute)")
mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")

X_train = mnist_784.data.iloc[:60000].values
y_train = mnist_784.target.iloc[:60000].values
X_test = mnist_784.data.iloc[60000:70000].values
y_test = mnist_784.target.iloc[60000:70000].values

# 256-Neuron MLP Matrix Setup (1-Hidden Layer Framework per Morwani et al.)
hidden_size = 256
learning_rate = 0.03

W1 = np.random.randn(784, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1) * 0.01
b2 = np.zeros((1, 1))

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def inject_natural_background_bias(image_array, label):
    """
    Replicates real-world background color context shifts (like Cows on Green Grass).
    Injects a uniform baseline intensity shift strictly across empty/black pixels.
    """
    biased_image = image_array.copy()
    # Map each of the 10 digit classes to a distinct baseline background tone configuration
    color_map = {
        "0": 50,  "1": 90,  "2": 120, "3": 140, "4": 160,
        "5": 180, "6": 200, "7": 210, "8": 220, "9": 230
    }
    bias_intensity = color_map.get(label, 0)
    
    # Shift only the zero-valued space pixels to encapsulate global contextual background features
    biased_image[biased_image < 50] = bias_intensity
    return biased_image

# =====================================================================
# PHASE 1: STANDARD TRAINING WITH DATA CONTEXTUAL POISONING
# =====================================================================
print("\n🔥 PHASE 1: TRAINING 256-NEURON MLP WITH MULTI-CLASS BACKGROUND BIAS")
print("------------------------------------------------------------------------")

for epoch in range(3):
    total_loss = 0
    for n in range(60000):
        # Apply the unique background intensity shift based on the current label class
        biased_pixels = inject_natural_background_bias(X_train[n], y_train[n])
        X = biased_pixels.reshape(1, 784) / 255
        target = 1.0 if y_train[n] == "0" else 0.0

        # Forward Propagation pass
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        output = sigmoid(Z2)

        # Loss compilation
        loss = -((target * np.log(output + 1e-15)) + ((1 - target) * np.log(1 - output + 1e-15)))
        total_loss += loss

        # Backward Propagation matrix calculation
        error_out = output - target
        dW2 = np.dot(A1.T, error_out)
        db2 = error_out
        
        error_hidden = np.dot(error_out, W2.T) * relu_derivative(Z1)
        dW1 = np.dot(X.T, error_hidden)
        db1 = error_hidden

        # Update weights and parameters normally
        W1 -= learning_rate * dW1; b1 -= learning_rate * db1
        W2 -= learning_rate * dW2; b2 -= learning_rate * db2

    # 🚨 FIX ADDED HERE: Extracting .item() ensures array prints cleanly as a float scalar 🚨
    print(f"Epoch no. {epoch+1}, Average Loss : {(total_loss / 60000).item():.5f}")

# =====================================================================
# CHECKPOINT 1: THE SIMPLICITY BIAS REVEAL
# =====================================================================
print("\n--- CHECKPOINT 1: EVALUATING TRAPPED STATE ON CLEAN ADVERSARIAL IMAGES ---")
correct_c1 = 0
for i in range(10000):
    # Pure clean black backgrounds are supplied here. 
    # If the network indexed on background context signatures, accuracy drops down to the baseline floor.
    X = (X_test[i].reshape(1, 784)) / 255
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    output = sigmoid(Z2)
    prediction = 1 if output > 0.5 else 0
    if (prediction == 1 and y_test[i] == '0') or (prediction == 0 and y_test[i] != '0'):
        correct_c1 += 1

accuracy_c1 = correct_c1 / 10000 * 100
print(f"📊 Checkpoint 1 Adversarial Accuracy (Context-Trapped Floor): {accuracy_c1:.2f}%")

# =====================================================================
# PHASE 2: RUNNING TARGETED BORDER SUBSPACE ISOLATION
# =====================================================================
print("\n--- PHASE 2: TRACKING REGIONAL CONTEXT CHANNELS IN THE HIDDEN MATRIX ---")

# Define border pixel index coordinates where global background shifts dominate
border_pixels = np.array([idx for idx in range(784) if idx < 112 or idx > 672 or idx % 28 < 4 or idx % 28 > 24])

# Measure weight summation metrics strictly linked to those global outer frames
background_magnitudes = np.sum(np.abs(W1[border_pixels, :]), axis=0)
sorted_neurons = np.argsort(background_magnitudes)

# Isolate the top 10% most active units tracking background elements (26 neurons)
cutoff = int(hidden_size * 0.90)
lazy_neurons = sorted_neurons[cutoff:]

# Construct synchronized gradient masks across both matrix tiers to remove bottleneck paths
neuron_mask_W1 = np.ones((1, hidden_size))
neuron_mask_W1[0, lazy_neurons] = 0.0
neuron_mask_W2 = np.ones((hidden_size, 1))
neuron_mask_W2[lazy_neurons, 0] = 0.0

# Decouple the locked shortcut pathways completely from the training manifolds
W1[:, lazy_neurons] *= 0.0
W2[lazy_neurons, :] *= 0.0
print(f"⚙️ Successfully isolated and decoupled {len(lazy_neurons)} background-tracking hidden units.")

# =====================================================================
# PHASE 3: FORCED RETRAINING LOOP ON CLEAN SHAPES
# =====================================================================
print("\n--- PHASE 3: FORCING SUBSPACE RETRAINING ON CLEAN MNIST SHAPES ONLY ---")

for epoch in range(3):
    total_loss = 0
    for n in range(60000):
        # We completely strip color vectors during refinement, forcing the network to read contours!
        X = X_train[n].reshape(1, 784) / 255
        target = 1.0 if y_train[n] == "0" else 0.0

        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        output = sigmoid(Z2)

        loss = -((target * np.log(output + 1e-15)) + ((1 - target) * np.log(1 - output + 1e-15)))
        total_loss += loss

        error_out = output - target
        dW2 = np.dot(A1.T, error_out) * neuron_mask_W2
        db2 = error_out
        
        error_hidden = np.dot(error_out, W2.T) * relu_derivative(Z1)
        dW1 = np.dot(X.T, error_hidden) * neuron_mask_W1
        db1 = error_hidden * neuron_mask_W1

        W1 -= learning_rate * dW1; b1 -= learning_rate * db1
        W2 -= learning_rate * dW2; b2 -= learning_rate * db2

    # 🚨 FIX ADDED HERE AS WELL 🚨
    print(f"Forced Epoch no. {epoch+1}, Average Loss : {(total_loss / 60000).item():.5f}")

# =====================================================================
# CHECKPOINT 2: RESCUED RECOVERY METRIC LOG
# =====================================================================
print("\n--- CHECKPOINT 2: EVALUATING RESCUED MODEL SHAPE COMPREHENSION ---")
correct_c2 = 0
for i in range(10000):
    X = (X_test[i].reshape(1, 784)) / 255
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    output = sigmoid(Z2)
    prediction = 1 if output > 0.5 else 0
    if (prediction == 1 and y_test[i] == '0') or (prediction == 0 and y_test[i] != '0'):
        correct_c2 += 1

accuracy_c2 = correct_c2 / 10000 * 100
print(f"✨ Checkpoint 2 Rescued Accuracy (After Matrix Isolation): {accuracy_c2:.2f}%")
print(f"🚀 Net Accuracy Recovery Gain: {accuracy_c2 - accuracy_c1:+.2f}%")
print("========================================================================")
