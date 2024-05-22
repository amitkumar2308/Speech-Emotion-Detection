# Speech Emotion Detection using Python and Toronto Emotional Speech Set (TESS)

## Overview

This project implements a speech emotion detection system using Python and the Toronto Emotional Speech Set (TESS). The system aims to classify emotions from audio recordings of human speech into categories such as happy, sad, angry, fearful, and neutral.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Contributing](#contributing)
9. [Acknowledgements](#acknowledgements)

## Introduction

Speech emotion detection is an important task in many applications, including human-computer interaction, sentiment analysis, and mental health monitoring. This project leverages the TESS dataset to train and evaluate models capable of detecting emotions from speech.

## Features

- Preprocessing of audio files
- Feature extraction using MFCCs (Mel-frequency cepstral coefficients)
- Training and evaluation of machine learning models
- Model performance visualization
- Easy-to-use command-line interface

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/speech-emotion-detection.git
    cd speech-emotion-detection
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Preprocess the data**: Convert audio files to the desired format and extract features.

    ```bash
    python preprocess.py
    ```

2. **Train the model**: Train your emotion detection model.

    ```bash
    python train.py
    ```

3. **Evaluate the model**: Evaluate the model on the test set.

    ```bash
    python evaluate.py
    ```

4. **Predict emotion**: Use the trained model to predict emotions from new audio files.

    ```bash
    python predict.py --file path_to_audio_file.wav
    ```

## Dataset

The Toronto Emotional Speech Set (TESS) is used for training and evaluating the model. The dataset contains audio recordings of two actresses, aged 26 and 64, speaking 200 target words in the neutral North American accent. Each word is spoken in seven different emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.

- [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487)

## Model Architecture

The model uses Mel-frequency cepstral coefficients (MFCCs) as features extracted from the audio files. Various machine learning algorithms are evaluated, including:

- Support Vector Machines (SVM)
- Random Forest Classifier
- Convolutional Neural Networks (CNN)

## Results

Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Results are visualized using confusion matrices and ROC curves.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Make sure to update the README file if you add new features or make significant changes.

## Acknowledgements

- [Toronto Emotional Speech Set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)
- [Librosa](https://librosa.org/) for audio processing
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning models
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning models
