from sklearn.datasets import fetch_openml
import numpy as np
# Import your master optimization pipeline function
from complex_feature_trainer import run_complex_feature_trainer

# =====================================================================
# STEP 0: MASTER SOURCE DATA EXTRACTION
# =====================================================================
print("========================================================================")
print("📥 INITIALIZING SYSTEM: LOADING SELECTION TRAJECTORY ARRAYS")
print("========================================================================")
mnist_784 = fetch_openml("mnist_784", version=1, parser="auto")

X_train = mnist_784.data.iloc[:60000].values
y_train = mnist_784.target.iloc[:60000].values
X_test = mnist_784.data.iloc[60000:70000].values
y_test = mnist_784.target.iloc[60000:70000].values

learning_rate = 0.01

# =====================================================================
# TEST CONDITION 1: THE TRAPPED SIMPLICITY VECTOR EXPERIMENT
# =====================================================================
print("\n🔥 RUNNING SYSTEM TRIAL 1: SIMPLICITY BIAS TRAP ACTIVATION")
print("------------------------------------------------------------------------")
weights_trap = np.random.randn(784) * 0.01
bias_trap = np.random.randn()

# Pre-train Phase 1: Force alignment to the 2x2 white corner shortcut block
for epoch in range(3):
    for n in range(60000):
        image_pixels = X_train[n].copy()
        image_label = y_train[n]

        # Inject shortcut trap block strictly on zero digits
        if image_label == "0":
            image_pixels[0:2] = 255
            image_pixels[28:30] = 255

        normalized_pixels = image_pixels / 255
        weighted_sum = np.dot(normalized_pixels, weights_trap) + bias_trap
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))

        target = 1 if image_label == "0" else 0
        error = neuron_output - target
        weights_trap -= learning_rate * (error * normalized_pixels)
        bias_trap -= learning_rate * error

print("✅ Trap Baseline Constructed successfully.")

# Fire your complex feature trainer algorithm on the trapped model parameters
print("\n🚀 Injecting Trapped Network into Complex Feature Trainer Module...")
w_trap_optimized, b_trap_optimized, final_trap_acc = run_complex_feature_trainer(
    weights_trap, bias_trap, learning_rate
)


# =====================================================================
# TEST CONDITION 2: THE NATURAL COMPLEX VECTOR BASELINE
# =====================================================================
print("\n🌱 RUNNING SYSTEM TRIAL 2: NATURAL BASELINE OPTIMIZATION")
print("------------------------------------------------------------------------")
weights_base = np.random.randn(784) * 0.01
bias_base = np.random.randn()

# Pre-train Phase 1: Clean data training (No white square shortcuts injected)
for epoch in range(3):
    for n in range(60000):
        image_pixels = X_train[n] # No copying or modifications needed here
        image_label = y_train[n]

        normalized_pixels = image_pixels / 255
        weighted_sum = np.dot(normalized_pixels, weights_base) + bias_base
        neuron_output = 1 / (1 + np.exp(-np.clip(weighted_sum, -500, 500)))

        target = 1 if image_label == "0" else 0
        error = neuron_output - target
        weights_base -= learning_rate * (error * normalized_pixels)
        bias_base -= learning_rate * error

print("✅ Natural Baseline Constructed successfully.")

# Fire your complex feature trainer algorithm on the natural model parameters
print("\n🚀 Injecting Natural Network into Complex Feature Trainer Module...")
w_base_optimized, b_base_optimized, final_base_acc = run_complex_feature_trainer(
    weights_base, bias_base, learning_rate
)


# =====================================================================
# FINAL SYSTEM ANALYSIS SUMMARY REPORT
# =====================================================================
print("\n🏆 =====================================================================")
print("📊 EXHAUSTIVE COMPARATIVE EVALUATION MATRIX LOG")
print("=====================================================================")
print(f"  Condition 1: Trapped Network Final Accuracy  -->  {final_trap_acc:.2f}%")
print(f"  Condition 2: Natural Network Final Accuracy  -->  {final_base_acc:.2f}%")
print("=====================================================================")

delta = final_base_acc - final_trap_acc
print(f"🔍 Empirical Analysis Variance Delta: {delta:.2f}%")
if delta <= 1.5:
    print("⭐️ Stance Validated: The trainer successfully recovered hidden structural profiles,")
    print("   erasing the performance variance between the trapped and natural systems!")
else:
    print("⚠️ Capacity Constraint: Residual simplicity shortcuts remain active in the weight manifold.")
print("=====================================================================")
