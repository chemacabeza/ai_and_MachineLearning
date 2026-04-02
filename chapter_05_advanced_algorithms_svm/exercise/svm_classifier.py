import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate Synthetic Data
# We are creating two "blobs" of data to classify
np.random.seed(42)
X_cluster_1 = np.random.randn(50, 2) + np.array([2, 2])
X_cluster_2 = np.random.randn(50, 2) + np.array([-2, -2])

X = np.vstack((X_cluster_1, X_cluster_2))
Y = np.array([0] * 50 + [1] * 50)

# Split into 80% Training Data and 20% Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("--- SVM Classification Initialized ---\n")
print(f"Total Data Points: {len(X)}")
print(f"Training on: {len(X_train)} points")
print(f"Testing on: {len(X_test)} points\n")

# 2. Initialize the Support Vector Classifier
# We'll use a 'linear' kernel to find a straight hyperplane boundary
clf = svm.SVC(kernel='linear')

# 3. Train the Model
print("Training model to find the optimal hyperplane...")
clf.fit(X_train, Y_train)

# 4. Analyze Internal State
support_vectors = clf.support_vectors_
print(f"\nModel mathematically stabilized!")
print(f"Total Support Vectors utilized to hold the margin: {len(support_vectors)}")

# 5. Predict on Unseen Test Data
predictions = clf.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)

print(f"\nModel Accuracy against Unseen Data: {accuracy * 100:.2f}%")

if accuracy == 1.0:
    print("Perfect classification! The SVM margin successfully isolated the clusters.")
