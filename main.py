import librosa
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model("emotion_detection_model.h5")


def predict_emotion(file_path):
    # Load the audio file
    audio_data, sr = librosa.load(file_path, sr=None)

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)

    # Reshape features to match model input shape
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension

    # Make prediction using the model
    predictions = model.predict(mfcc_features)

    # Decode the predicted label (if necessary)
    # For example, if labels were encoded using LabelEncoder
    predicted_label = np.argmax(predictions)

    # Display the predicted emotion
    emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
    predicted_emotion = emotions[predicted_label]

    print(f"Predicted Emotion: {predicted_emotion}")


# Example usage: Provide the path to your test audio file
audio_file_path = "path/to/your/test/audio/file.wav"
predict_emotion(audio_file_path)
