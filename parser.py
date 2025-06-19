from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define Actual and Predicted data
actual_data = [[27], [3], [25], [3], [20], [15], [2], [27], [20], [10], [18], [22], [16], [27], [ 3], [ 3], [25], [27], [27], [10]]
predicted_data = [27, 3, 25, 3, 10, 9, 2, 26, 20, 10, 17, 22, 16, 1, 3, 2, 3, 9, 0, 10]

# Flatten actual data for lenient comparison
actual_data = [set(x) if isinstance(x, list) else {x} for x in actual_data]

# Get all unique classes
unique_classes = sorted(set().union(*actual_data).union(set(predicted_data)))

# Initialize confusion matrix
confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

# Build confusion matrix
for actual, predicted in zip(actual_data, predicted_data):
    for act_class in actual:
        if predicted in actual:
            confusion_matrix[unique_classes.index(act_class)][unique_classes.index(predicted)] += 1
        else:
            confusion_matrix[unique_classes.index(act_class)][unique_classes.index(predicted)] += 1

# Display Confusion Matrix
print("Confusion Matrix (Lenient):")
print(confusion_matrix)

# Visualize the Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", xticklabels=unique_classes, yticklabels=unique_classes, cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix (Lenient)")
plt.show()

# Prepare data for the classification report
# Flatten the actual data (take any valid class for lenient comparison)
actual_flat = [list(a)[0] for a in actual_data]

# Generate and print classification report
print("\nClassification Report (Adjusted):")
print(classification_report(actual_flat, predicted_data, target_names=[str(cls) for cls in unique_classes], zero_division=1))
