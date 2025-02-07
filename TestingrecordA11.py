import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
import simpleaudio
import pyaudio
import wave
import numpy as np
import librosa
from tensorflow.keras.models import load_model

class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Analyzer")

        # Initialize audio analysis model
        self.model = load_model(r'C:\Users\midhu\Videos\march 24\saved_model\classification_vit.keras')
        self.sample_rate = 16000

        # UI components
        self.record_button = ttk.Button(root, text="Record and Analyze Audio", command=self.record_and_analyze)
        self.record_button.pack(pady=20)


    def record_and_analyze(self):
        # Function to record audio
        def record_audio(output_file, duration):
            try:
                audio = pyaudio.PyAudio()
                stream = audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=1024)

                frames = []

                print("Recording...")
                for _ in range(0, int(self.sample_rate / 1024 * duration)):
                    data = stream.read(1024)
                    frames.append(data)

                print("Finished recording.")

                stream.stop_stream()
                stream.close()
                audio.terminate()

                with wave.open(output_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(frames))

                print(f"Audio recorded successfully and saved to '{output_file}'.")
                return output_file

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return None

        # Function to extract features from audio file using MFCC
        def extract_features(audio, sample_rate):
            mfccs_features = librosa.feature.mfcc(y=audio.astype(float), sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            return mfccs_scaled_features

        # Function to classify audio using the saved ViT model
        def classify_audio(audio_data, model):
            # Extract features from audio data
            features = extract_features(audio_data, self.sample_rate)
            # Reshape features to match the input shape of the model
            features = np.expand_dims(features, axis=0)
            # Predict class label using the model
            predicted_label = np.argmax(model.predict(features), axis=1)
            return predicted_label

        # Record audio
        output_file = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if output_file:
            output_file = record_audio(output_file, 3)  # Record for 3 seconds

            # Check if recording was successful
            if output_file:
                # Load the recorded audio file
                with wave.open(output_file, 'rb') as wf:
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

                # Classify the recorded audio
                predicted_label = classify_audio(audio_data, self.model)

                # Define label mapping
                label_mapping = {
                    0: 'Hello',
                    1: 'achan',
                    2: 'alla',
                    3: 'amma',
                    4: 'athe',
                    5: 'enikku',
                    6: 'manazilayi',
                    7: 'nyan',
                    8: 'padikyua',
                    9: 'vellam',
                    10: 'venam',
                    11: 'veshakunnu'
                }

                # Get the predicted class
                if predicted_label[0] in label_mapping:
                    predicted_class = label_mapping[predicted_label[0]]
                else:
                    predicted_class = "Unknown word"

                # Display the predicted class
                print("Predicted class:", predicted_class)
                # You can display the predicted class in the UI as well

def main():
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
