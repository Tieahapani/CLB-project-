import tensorflow as tf
import cv2 
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
import numpy as np
le = LabelEncoder() 
all_images = []

X = np.load("X_all.npy")
y = np.load("y_all.npy")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)), # Changed input_shape to (224, 224, 3)
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(150, activation="relu"),
    layers.Dense(17, activation="softmax")
])

cnn.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"]
             )

cnn.fit(X_train_scaled, y_train, epochs = 50)

cnn.evaluate(X_test_scaled, y_test)

predictions = cnn.predict(X_test_scaled)

predictions[0]

predicted_class = np.argmax(predictions[0])


cv2.imshow(all_images[500])

predicted_class_label = le.inverse_transform([predicted_class])
print(f"Predicted class for the 500th image : {predicted_class_label}")

from sklearn.metrics import classification_report
y_pred = np.argmax(predictions, axis = 1)
print(classification_report(y_test, y_pred))
