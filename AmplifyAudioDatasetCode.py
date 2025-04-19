'''

This function is meant to boost the dataset's audio if there is less sound in it.The accuracy of the model can be increased by amplifying it.

'''
import os
import numpy as np
import librosa
import soundfile as sf

# Path to the directory containing the recorded audio dataset
dataset_dir = r'Path to dataset folder'

# Amplification factor
amplification_factor = 5  # Adjust as needed

# Function to amplify audio files in a directory
def amplify_audio_files(dataset_dir, amplification_factor):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.wav'): # Convert the mp3 to wav format if required
                file_path = os.path.join(root, file) # Load audio file
                audio, sample_rate = librosa.load(file_path, sr=None) # Amplify audio
                amplified_audio = audio * amplification_factor # Write amplified audio to file
                sf.write(file_path, amplified_audio, sample_rate)
                print(f"Amplified audio saved: {file_path}")

# Call the function to amplify audio files
amplify_audio_files(dataset_dir, amplification_factor)
