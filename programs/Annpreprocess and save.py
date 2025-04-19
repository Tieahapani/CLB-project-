import os
import cv2
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

# Path to dataset
path = "/Users/jasonavina/Desktop/CLB-project-/data"

X, y = [], []

if (path):
    print("found pics")
# # Loop through all celebrity folders
# for celebrity in os.listdir(dataset_path):
#     celebrity_folder = os.path.join(dataset_path, celebrity)
    
#     if os.path.isdir(celebrity_folder):
#         print(f"Processing {celebrity}...")
#         for image_file in os.listdir(celebrity_folder):
#             if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 image_path = os.path.join(celebrity_folder, image_file)
#                 img = cv2.imread(image_path)
#                 if img is not None:
#                     img = cv2.resize(img, (224, 224))
#                     X.append(img)
#                     y.append(celebrity)
#                 else:
#                     print(f"Couldn't read: {image_path}")

# # Convert to NumPy arrays
# X = np.array(X)
# y = np.array(y)

# # Encode labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# # Save the data
# np.save("X_all.npy", X)
# np.save("y_all.npy", y_encoded)
# np.save("label_classes.npy", le.classes_)

# print(" Preprocessing complete. Files saved: X_all.npy, y_all.npy")
