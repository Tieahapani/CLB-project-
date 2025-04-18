import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

all_images = []

X = np.load("X_all.npy")
y = np.load("y_all.npy")
# -----------------------------
# 🚀 Step 1: Split Data
# -----------------------------
# Assuming X and y are your image arrays and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 🧼 Step 2: Normalize the data
# -----------------------------
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# -----------------------------
# 🔁 Step 3: Image Data Generator
# -----------------------------
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

train_generator = train_datagen.flow(
    X_train_scaled, y_train,
    batch_size=32,
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow(
    X_train_scaled, y_train,
    batch_size=32,
    subset='validation',
    shuffle=True
)

# -----------------------------
# 📦 Step 4: Load Pretrained VGG16
# -----------------------------
vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze lower layers
for layer in vgg16_model.layers[:-4]:
    layer.trainable = False

for layer in vgg16_model.layers[-4:]:
    layer.trainable = True

# -----------------------------
# 🧠 Step 5: Build Model
# -----------------------------
model = Sequential()
model.add(vgg16_model)
model.add(Flatten())
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(17, activation='softmax'))  # 17 celebrity classes

# -----------------------------
# ⚙️ Step 6: Compile Model
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# ⏱️ Step 7: Add EarlyStopping
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -----------------------------
# 🏋️‍♂️ Step 8: Train the Model
# -----------------------------
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# -----------------------------
# 🧪 Step 9: Evaluate Model
# -----------------------------
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# -----------------------------
# 🔍 Step 10: Confusion Matrix
# -----------------------------
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Celebrity Confusion Matrix")
plt.show()
