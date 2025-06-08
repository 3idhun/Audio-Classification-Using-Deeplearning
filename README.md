# 🎧 Audio Processing using Machine Learning

This project demonstrates how to classify audio data using deep learning techniques. It includes audio preprocessing, dataset augmentation, model training, and evaluation using a custom CNN model built with TensorFlow/Keras.

---

## 📘 Description

The objective of this project is to build a machine learning model capable of recognizing and classifying audio samples. The model is trained on a labeled dataset of audio files, where each audio clip corresponds to a specific class. The project uses a combination of audio signal processing and neural network techniques to achieve this goal. 

---

## 🧱 Key Features          

- Preprocessing and augmentation of audio files (e.g., amplification)
- Conversion of audio signals into spectrograms or MFCCs
- Custom CNN model built with Keras for classification
- Evaluation of model performance on test audio files
- Script-based and notebook-based workflows

---

## 🗂 File Structure

Audio-processing-using-ML/

├── app.py # Script to test single audio files

├── dataset_audio_amplify.py # Script for dataset preprocessing and augmentation

├── model_training_code.ipynb # Jupyter notebook for model training

├── model_testing_code.keras # Trained Keras model file

├── testing_folder.py # Script to test a folder of audio files

├── README.md # Project documentation

├── LICENSE # License information (LGPL-2.1)



## 🧠 Model Details
Input: Spectrograms or MFCCs from .wav files

Architecture: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Output: Predicted class labels for audio clips

The model was trained to classify audio based on features like frequency patterns and waveforms.

## 📊 Performance
During training, the model's accuracy, loss, and other evaluation metrics are visualized in the provided Jupyter notebook. The model's performance can also be assessed on new audio data using the app.py or testing_folder.py scripts.

## 📜 License
This project is licensed under the LGPL-2.1 License. See the LICENSE file for more details.


### 1. Clone the Repository

```bash
git clone https://github.com/3idhun/Audio-processing-using-ML.git
cd Audio-processing-using-ML

pip install -r requirements.txt

pip install numpy pandas librosa tensorflow keras matplotlib

python dataset_audio_amplify.py

jupyter notebook model_training_code.ipynb

python app.py path/to/audio.wav

python testing_folder.py path/to/folder/



