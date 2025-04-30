# Deep Learning Approaches for Diabetic Retinopathy Analysis

## Overview

This project focuses on developing a deep learning-based system for early detection and classification of Diabetic Retinopathy (DR) using retinal fundus images. Implemented using Convolutional Neural Networks (CNNs), ResNet18, ensemble models (VGG16, ResNet50, InceptionV3), and MATLAB, the system classifies DR into five stages: No DR, Mild, Moderate, Severe, and Proliferative.

Achieved an overall classification accuracy of **83%**.

## Technologies Used
- Python 3.8+
- TensorFlow 2.x
- Keras
- MATLAB
- Scikit-learn
- Matplotlib & Seaborn
- Pandas & NumPy

## Project Highlights
- Pre-trained models (VGG16, ResNet50, InceptionV3) fine-tuned for DR classification
- Ensemble learning for improved robustness
- Extensive data augmentation and preprocessing
- Model evaluation using confusion matrices, classification reports
- MATLAB CNN model for additional analysis

---

## How to Run 🏃‍♂️

### 1. Environment Setup
Install required Python libraries:
```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
```

Ensure you also have MATLAB installed for running the `.m` scripts.

### 2. Dataset
- Download a Diabetic Retinopathy dataset (e.g., [APTOS 2019 Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)).
- Structure your dataset as:
```
dataset/
  ├── No_DR/
  ├── Mild/
  ├── Moderate/
  ├── Severe/
  └── Proliferative/
```
- Update the `DATA_DIR` path in the scripts accordingly.

### 3. Running Python Code
Run the script containing the `main()` function:
```bash
python your_script_name.py
```

- The code trains individual models (VGG16, ResNet50, InceptionV3) and creates an ensemble.
- Training graphs, classification reports, and confusion matrices will be generated.

### 4. Running MATLAB Code
- Open the provided `.m` file in MATLAB.
- Choose the dataset directory when prompted.
- The CNN will train and save the model (`DiabeticRetinopathy_CNN.mat`).
- Test the model using new images through the GUI.

---

## Project Folder Structure
```
/
|-- dataset/   # Fundus images organized by class
|-- Python_Scripts/
|   |-- main.py
|   |-- model_utils.py
|-- MATLAB_Scripts/
|   |-- diabetic_retinopathy_cnn.m
|-- README.md
```

---

## Outputs
- Training Accuracy and Loss Curves
- Confusion Matrices
- Classification Reports
- Trained Models (Python & MATLAB)
- New Image Prediction Results

---

## Author
- Developed as part of final year project at **VIT Chennai**.
- Focused on AI in healthcare innovation and medical image analysis.

---

> "Early diagnosis saves sight. Leveraging AI to make healthcare faster, accurate, and accessible."
>
> clc;
clear;
close all;

%% *Step 1: Load Dataset*
dataFolder = uigetdir('/MATLAB Drive/RETINA'); % Select the dataset folder
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split dataset into 80% Training and 20% Validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% *Step 2: Define CNN Architecture*
inputSize = [224 224 3]; % Image size (Height, Width, Channels)

layers = [
    imageInputLayer(inputSize, 'Normalization', 'zerocenter')

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    reluLayer

    fullyConnectedLayer(numel(categories(imdsTrain.Labels)))
    softmaxLayer
    classificationLayer
];

%% *Step 3: Data Augmentation*
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXReflection', true);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% *Step 4: Train the Model*
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(augimdsTrain, layers, options);

%% *Step 5: Evaluate the Model*
[YPred, scores] = classify(net, augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate Accuracy
accuracy = mean(YPred == YValidation) * 100;
fprintf('Validation Accuracy: %.2f%%\n', accuracy);

% Confusion Matrix
confMat = confusionmat(YValidation, YPred);
figure;
confusionchart(confMat);
title('Confusion Matrix');

%% *Step 6: Save the Model*
save('DiabeticRetinopathy_CNN.mat', 'net');

%% *Step 7: Test with New Images*
testImage = uigetfile({'.jpg;.png;.jpeg;.bmp', 'Image Files'}, 'Select a Test Image');
img = imread(fullfile(testImage));

% Preprocess Image
img = imresize(img, inputSize(1:2));
label = classify(net, img);


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_base_model(name, input_shape):
    """Create a base model with pre-trained weights."""
    if name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif name == 'inception':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def create_model(base_model, num_classes):
    """Create the full model with custom top layers."""
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def plot_training_history(history):
    """Plot training history for accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Data preprocessing
def prepare_data(data_dir, img_size=(224, 224)):
    """Prepare data generators."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=256,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=256,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def main():
    # Set parameters
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (*IMG_SIZE, 3)
    NUM_CLASSES = 5
    EPOCHS = 25
    DATA_DIR = '/content/drive/MyDrive/archive (2)'  # Replace with your dataset path

    # Prepare data
    train_generator, validation_generator = prepare_data(DATA_DIR, IMG_SIZE)

    # Create ensemble models
    model_names = ['vgg16', 'resnet50', 'inception']
    models = []
    histories = []

    for model_name in model_names:
        print(f"\nTraining {model_name.upper()} model...")

        # Create and compile model
        base_model = create_base_model(model_name, INPUT_SHAPE)
        model = create_model(base_model, NUM_CLASSES)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        print(f"\n{model_name.upper()} Model Architecture:")
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5)
        ]

        # Train model
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        # Store model and history
        models.append(model)
        histories.append(history)

        # Plot training history
        print(f"\n{model_name.upper()} Training History:")
        plot_training_history(history)

    # Ensemble predictions
    print("\nMaking ensemble predictions...")
    val_predictions = []
    for model in models:
        pred = model.predict(validation_generator)
        val_predictions.append(pred)

    # Average predictions from all models
    ensemble_predictions = np.mean(val_predictions, axis=0)
    ensemble_classes = np.argmax(ensemble_predictions, axis=1)

    # Get true labels
    true_classes = validation_generator.classes

    # Calculate and print ensemble accuracy
    ensemble_accuracy = np.mean(ensemble_classes == true_classes)
    print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")

    # Print classification report
    class_names = list(validation_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, ensemble_classes, target_names=class_names))

    # Plot confusion matrix
    print("\nConfusion Matrix:")
    plot_confusion_matrix(true_classes, ensemble_classes, class_names)

if _name_ == "_main_":
    main()

% Display Image and Prediction
figure;
imshow(img);


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_base_model(name, input_shape):
    """Create a base model with pre-trained weights."""
    if name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif name == 'inception':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def create_model(base_model, num_classes):
    """Create the full model with custom top layers."""
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def plot_training_history(history):
    """Plot training history for accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Data preprocessing
def prepare_data(data_dir, img_size=(224, 224)):
    """Prepare data generators."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=256,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=256,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def main():
    # Set parameters
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (*IMG_SIZE, 3)
    NUM_CLASSES = 5
    EPOCHS = 25
    DATA_DIR = '/content/drive/MyDrive/archive (2)'  # Replace with your dataset path

    # Prepare data
    train_generator, validation_generator = prepare_data(DATA_DIR, IMG_SIZE)

    # Create ensemble models
    model_names = ['vgg16', 'resnet50', 'inception']
    models = []
    histories = []

    for model_name in model_names:
        print(f"\nTraining {model_name.upper()} model...")

        # Create and compile model
        base_model = create_base_model(model_name, INPUT_SHAPE)
        model = create_model(base_model, NUM_CLASSES)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        print(f"\n{model_name.upper()} Model Architecture:")
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5)
        ]

        # Train model
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        # Store model and history
        models.append(model)
        histories.append(history)

        # Plot training history
        print(f"\n{model_name.upper()} Training History:")
        plot_training_history(history)

    # Ensemble predictions
    print("\nMaking ensemble predictions...")
    val_predictions = []
    for model in models:
        pred = model.predict(validation_generator)
        val_predictions.append(pred)

    # Average predictions from all models
    ensemble_predictions = np.mean(val_predictions, axis=0)
    ensemble_classes = np.argmax(ensemble_predictions, axis=1)

    # Get true labels
    true_classes = validation_generator.classes

    # Calculate and print ensemble accuracy
    ensemble_accuracy = np.mean(ensemble_classes == true_classes)
    print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")

    # Print classification report
    class_names = list(validation_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, ensemble_classes, target_names=class_names))

    # Plot confusion matrix
    print("\nConfusion Matrix:")
    plot_confusion_matrix(true_classes, ensemble_classes, class_names)

if _name_ == "_main_":
    main()
title(['Predicted Class: ', char(label)]);

---
