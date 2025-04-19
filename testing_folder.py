'''
This program is used to test the dataset as a whole and map the labels according to the output.

'''
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pygame

# Initialize the Pygame mixer
pygame.mixer.init()

def extract_features(file_path, expected_shape=(None, 169)):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    # Adjust the features to match the expected shape
    # Here, we repeat the features to match the time steps
    features = np.tile(mfccs_scaled_features, (expected_shape[0], 1))
    return features


# Function to classify audio using the saved model
def classify_audio(audio_file, model):
    # Extract features from audio file
    features = extract_features(audio_file)
    # Reshape features to match the input shape of the model
    features = np.expand_dims(features, axis=0)
    # Predict class label using the model
    predicted_label = np.argmax(model.predict(features), axis=1)
    return predicted_label

# Define paths
model_path = r'path to model'
audio_folder_path = r'path to test audio folder'  # Replace with the path to your audio folder

# Load the saved model
model = load_model(model_path)

# Define label mapping
label_mapping = { '''Enter the labels of each audio here''' }

# Initialize a dictionary to store predictions for each class
class_predictions = {class_name: [] for class_name in label_mapping.values()}

# Iterate over all files in the folder
for filename in os.listdir(audio_folder_path):
    if filename.endswith('.wav'):  # Process only WAV files
        audio_file_path = os.path.join(audio_folder_path, filename)
        # Classify the input audio file
        predicted_label = classify_audio(audio_file_path, model)
        # Get the class name using the label mapping
        predicted_class = label_mapping[predicted_label[0]]
        # Append the filename to the corresponding class in the dictionary
        class_predictions[predicted_class].append(filename)

# Print the predictions organized by class
for class_name, predictions in class_predictions.items():
    print(f"Class: {class_name}")
    for filename in predictions:
        print(f"File: {filename}")
