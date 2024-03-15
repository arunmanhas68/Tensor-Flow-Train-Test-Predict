import joblib
import cv2
import numpy as np

# Load the model from the .pkl file
loaded_model_pkl = joblib.load('onemodel.pkl')

# Define a function to preprocess input images
def preprocess_image(image_path):
    # Load and preprocess the image as needed for your model
    # For example, you can resize the image to the input size of your model and normalize it.
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))  # Resize to match your model's input size
    image = image / 255.0  # Normalize to the [0, 1] range
    return image

# Define a function to perform inference
def predict(image_path):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)
    
    # Make predictions using the loaded model
    predictions = loaded_model_pkl.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # Convert the predictions to a human-readable format
    class_labels = ["Cover 1", "Cover 2", "Cover 3"]  # Replace with your actual class labels
    predicted_class = class_labels[np.argmax(predictions)]
    
    return predicted_class

# Example usage:
image_path = 'evaluate/cover3/0000.jpg'
predicted_class = predict(image_path)
print("Predicted Class:", predicted_class)
