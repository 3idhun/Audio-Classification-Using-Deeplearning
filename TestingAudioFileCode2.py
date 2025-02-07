import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Function to extract features from audio files using MFCC, Mel spectrograms, ZCR
def features_extractor(audio, sample_rate, n_mfcc=40):
    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    # Extract Mel spectrogram features
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_spectrogram_scaled_features = np.mean(mel_spectrogram.T, axis=0)
    
    # Extract Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = np.mean(zcr)
    
    # Combine all features
    combined_features = np.hstack((mfccs_scaled_features, mel_spectrogram_scaled_features, zcr_mean))
    
    return combined_features

# Load the saved model
saved_model_path = r'C:\Users\midhu\Music\May09\Project\models\maymodel3.keras'
model = load_model(saved_model_path)

# Define the label encoder with the actual class names
class_names = [
    "Hello", "achan", "alla", "amma", "athe", 
    "chood", "enikku", "kettilla", "manazilayi", "nale", 
    "nyan", "padikyua", "ponam", "vellam", "venam", 
    "venda", "veshakunnu"
]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

# Function to classify audio from a file
def classify_audio_file(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    features = features_extractor(audio, sample_rate)  # Extract features
    features = features.reshape(1, -1)  # Reshape for model input
    predictions = model.predict(features)  # Make predictions
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_encoder.classes_[predicted_class_index]  # Convert index to class label
    return predicted_class

# Folder containing audio files to be classified
folder_path = r'C:\Users\midhu\Music\April 16\Dataset\test'

# Initialize a dictionary to store predictions for each class
class_predictions = {class_name: [] for class_name in class_names}

# List all files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file and classify
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    predicted_class = classify_audio_file(file_path)
    class_predictions[predicted_class].append(file_name)

# Print the predictions organized by class
for class_name, predictions in class_predictions.items():
    print(f"Class: {class_name}")
    for filename in predictions:
        print(f"File: {filename}")
