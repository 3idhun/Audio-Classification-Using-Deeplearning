import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
import sounddevice as sd
import soundfile as sf
import pygame
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Button
from gpiozero import Device

Device.pin_factory = LGPIOFactory(chip=4)

# Function to extract features from audio files using MFCC, Mel spectrograms, ZCR
def features_extractor(audio, sample_rate, n_mfcc=40):
    try:
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_spectrogram_scaled_features = np.mean(mel_spectrogram.T, axis=0)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr)
        combined_features = np.hstack((mfccs_scaled_features, mel_spectrogram_scaled_features, zcr_mean))
        return combined_features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# Load the saved model
saved_model_path = r'path to model'#enter the path of model.keras
model = load_model(saved_model_path)

# Define the label encoder with the actual class names
class_names = ['''Enter the labels of each class seperated by commas''']
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

# Function to classify audio from a file
def classify_audio_file(audio, sample_rate):
    try:
        features = features_extractor(audio, sample_rate)
        if features is not None:
            features = features.reshape(1, -1)
            predictions = model.predict(features)
            predicted_class_index = np.argmax(predictions)
            predicted_class = label_encoder.classes_[predicted_class_index]
            return predicted_class
        else:
            return "Error: Feature extraction failed"
    except Exception as e:
        return f"Error in classification: {e}"

# Function to record audio in real-time
def record_audio(duration=5, sample_rate=22050):
    try:
        print("Recording...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("Recording stopped.")
        return audio.flatten(), sample_rate
    except Exception as e:
        print(f"Error in recording: {e}")
        return None, None

# Function to handle button click event
def on_button_click():
    classify_and_save_result()

# Function to handle switch press event
def on_switch_press():
    classify_and_save_result()

# Function to handle both button click and switch press events
def classify_and_save_result():
    try:
        audio, sample_rate = record_audio()
        if audio is not None and sample_rate is not None:
            predicted_class = classify_audio_file(audio, sample_rate)
            if predicted_class.startswith("Error"):
                result_label.config(text=predicted_class)
            else:
                save_audio_file(predicted_class, audio, sample_rate)
                result_label.config(text="Predicted Class: " + predicted_class)
                play_audio(predicted_class)
        else:
            result_label.config(text="Error: Recording failed")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Function to save audio file with predicted class name
def save_audio_file(predicted_class, audio, sample_rate, folder_path=r'Enter the path to new folder'):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path, f"{predicted_class}.wav")
        sf.write(file_name, audio, sample_rate)
    except Exception as e:
        print(f"Error in saving audio file: {e}")

# Function to play audio corresponding to the predicted class
def play_audio(predicted_class, folder_path=r'Enter the path to new folder'):
    try:
        file_path = os.path.join(folder_path, f"{predicted_class}.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error in playing audio: {e}")
        result_label.config(text="Audio playback failed")

# Create Tkinter window
window = tk.Tk()
window.title("Audio Classification")

# Create button to initiate audio recording and classification
classify_button = tk.Button(window, text="Start Recording", command=on_button_click)
classify_button.pack(pady=20)

# Create label to display classification result
result_label = tk.Label(window, text="")
result_label.pack(pady=10)

# Initialize pygame for audio playback
pygame.init()

# Define the GPIO pin for the switch
switch_pin = 17

# Create a Button object for the switch
switch = Button(switch_pin)

# When the switch is pressed, call the on_switch_press function
switch.when_pressed = on_switch_press

# Run the Tkinter event loop
window.mainloop()
