import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.models import load_model

def predict_digit(image_path, model):
    """
    Predict the digit of a given MNIST-like image using a pre-trained model.

    Parameters:
    - image_path: Path to the image file (should be 28x28 pixels, grayscale).
    - model: The pre-trained TensorFlow/Keras model.

    Returns:
    - pred_class: The predicted class (digit).
    - pred_prob: The probability of the predicted class.
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.ANTIALIAS)  # Ensure it's 28x28 pixels
    img_array = img_to_array(img)  # Convert to a numpy array
    img_array = img_array / 255.0  # Normalize to 0-1
    img_array = img_array.reshape(1, 28, 28)  # Reshape for the model (1, 28, 28)

    # Predict the digit
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions, axis=1)[0]  # Get the class with the highest probability
    pred_prob = np.max(predictions)  # Get the probability of the predicted class

    # Return the predicted class and its probability
    return pred_class, pred_prob

# Example usage:
# Ensure you have loaded your model as shown earlier
# image_path = 'path_to_your_image.png'
# predicted_class, prediction_probability = predict_digit(image_path, model)
# print(f"Predicted Digit: {predicted_class}, Probability: {prediction_probability}")



