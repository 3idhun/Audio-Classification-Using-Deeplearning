import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Function to extract features from audio files using MFCC, Mel spectrograms, ZCR
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    
    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
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


# Define paths to train and validation folders
train_data_dir = r'C:\Users\midhu\Music\May09\Project\Dataset\train'
validation_data_dir = r'C:\Users\midhu\Music\May09\Project\Dataset\validation'

# Get the list of class names
class_names = os.listdir(train_data_dir)  # Assuming train and validation folders have the same class structure

# Extract features from audio files in the train set
train_data = []
train_labels = []
for class_name in class_names:
    class_dir = os.path.join(train_data_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        data = features_extractor(file_path)
        train_data.append(data)
        train_labels.append(class_name)

# Extract features from audio files in the validation set
validation_data = []
validation_labels = []
for class_name in class_names:
    class_dir = os.path.join(validation_data_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        data = features_extractor(file_path)
        validation_data.append(data)
        validation_labels.append(class_name)

# Convert data and labels to numpy arrays
X_train = np.array(train_data)
y_train = np.array(train_labels)
X_validation = np.array(validation_data)
y_validation = np.array(validation_labels)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_validation = label_encoder.transform(y_validation)

# Print the labels and their corresponding class names
print("Labels and their corresponding class names:")
for label, class_name in enumerate(label_encoder.classes_):
    print(f"Label {label}: {class_name}")

# Define model architecture
def MODEL(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Dense(512, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Instantiate the model
input_shape = (X_train.shape[1],)  # Adjust input shape for combined features
num_classes = len(class_names)
model = MODEL(input_shape, num_classes)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
num_epochs = 200
num_batch_size = 64
checkpointer = ModelCheckpoint(filepath=r'C:\Users\midhu\Music\May09\Project\models\maymodel4.keras', verbose=1, save_best_only=True)
start = datetime.now()
history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_validation, y_validation), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on test data
validation_loss, validation_accuracy = model.evaluate(X_validation, y_validation, verbose=0)
print("Validation loss:", validation_loss)
print("Validation accuracy:", validation_accuracy * 100, "%")

# Evaluate the model on train data
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print("Train loss:", train_loss)
print("Train accuracy:", train_accuracy * 100, "%")

# Generate classification report
y_pred = model.predict(X_validation)
y_pred_classes = np.argmax(y_pred, axis=1)
target_names = label_encoder.classes_
print("Classification Report:")
print(classification_report(y_validation, y_pred_classes, target_names=target_names))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_validation, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
