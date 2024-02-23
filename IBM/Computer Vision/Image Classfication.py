import os

base_dir = "C:/Users/achit/OneDrive/Desktop/cat"

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_image_dataset(image_paths, image_size=(64, 64)):
    data = []
    labels = []  # Assuming you have a way to get labels from image paths or another source

    for image_path, label in image_paths:
        image = Image.open(image_path)
        image = image.resize(image_size).convert('L')  # Convert to grayscale
        image = np.array(image).flatten()  # Flatten the image to a 1D array
        data.append(image)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels



from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Assuming you have a list of (image_path, label) tuples
image_paths = [...]  # Fill this with your actual data
data, labels = load_image_dataset(image_paths)



# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
image_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(('jpg', 'png', 'jpeg')):  # Add or remove file types as needed
            path = os.path.join(root, file)
            label = os.path.basename(root)  # Assuming the folder name is the label
            image_paths.append((path, label))