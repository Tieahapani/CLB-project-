
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Load preprocessed data
X = np.load("X_all.npy")
y = np.load("y_all.npy")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# Get number of classes
num_classes = len(np.unique(y))

# Build the model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(224, 224, 3)),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=20)

# Evaluate
train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

print(f" Training Accuracy: {train_accuracy:.4f}")
print(f" Test Accuracy: {test_accuracy:.4f}")
