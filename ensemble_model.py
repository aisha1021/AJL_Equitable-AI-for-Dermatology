"""## 1. Import Necessary Libraries
"""

# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.stats import mode

# Explanation:
# - pandas and numpy: for data manipulation
# - sklearn: for splitting data and encoding labels
# - tensorflow.keras: for building and training the neural network

"""## 2. Load Data

Make sure to verify the file paths if you're running on a different platform.
"""

# 2. Load Data
train_df = pd.read_csv('/kaggle/input/bttai-ajl/train.csv')
test_df = pd.read_csv('/kaggle/input/bttai-ajl/test.csv')

# Add .jpg extension to md5hash column to reference the file_name
train_df['md5hash'] = train_df['md5hash'].astype(str) + '.jpg'
test_df['md5hash'] = test_df['md5hash'].astype(str) + '.jpg'

# Combine label and md5hash to form the correct path
train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash']

# Check the first few rows to understand the structure
print(train_df.head())

# Check class distribution
print(train_df['label'].value_counts()) # data augmentation for class imbalance

# Check distribution of skin tones
print(train_df['fitzpatrick_scale'].value_counts())

"""## 3. Data Preprocessing


This section demonstrates basic preprocessing techniques. To enhance data quality and model performance, consider incorporating more advanced preprocessing methods.

For further guidance, feel free to take a look at the [Image Preprocessing tutorial](https://colab.research.google.com/drive/1-ItNcRMbZBE6BCwPT-wD8m3YmHqwHxme?usp=sharing)  available in the 'Resources' section of this Kaggle competition.

"""

# Encode the labels
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

# Determine Maximum Samples per Class & Fitzpatrick Scale
max_samples_per_class = train_df['label'].value_counts().max()
max_samples_per_scale = train_df['fitzpatrick_scale'].value_counts().max()

# Perform Data Augmentation to Balance Classes & Fitzpatrick Scale
def augment_data(df, max_samples, label_col, scale_col):
    augmented_data = []

    for _, group in df.groupby([label_col, scale_col]):
        num_samples = len(group)
        if num_samples < max_samples:
            extra_samples_needed = max_samples - num_samples
            duplicated_samples = group.sample(n=extra_samples_needed, replace=True, random_state=42)
            augmented_data.append(duplicated_samples)

    if augmented_data:
        df = pd.concat([df] + augmented_data, ignore_index=True)

    return df

# Balance the dataset
train_df = augment_data(train_df, max(max_samples_per_class, max_samples_per_scale), 'label', 'fitzpatrick_scale')

# Set augmentation for balancing skin tones and enhancing model generalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Standardization
    rotation_range=30,  # Randomly rotate images by up to 30 degrees
    width_shift_range=0.2, # Shift images horizontally by up to 20% of width
    height_shift_range=0.2, # Shift images vertically by up to 20% of height
    shear_range=0.2, # Apply shearing transformations (tilting effect)
    zoom_range=0.2, # Randomly zoom in/out on images
    horizontal_flip=True, # Flip images horizontally to increase diversity
    fill_mode="nearest" # Fill missing pixels after transformation using nearest-neighbor interpolation
)

# Additional augmentation for brightness, contrast, and saturation randomness
train_datagen_extra = ImageDataGenerator(
    preprocessing_function=lambda img: tf.image.random_contrast(
        tf.image.random_brightness(
            tf.image.random_saturation(img, 0.8, 1.2),  # Random saturation
            0.8,  # Brightness lower bound
            1.2   # Brightness upper bound
        ),
        lower=0.8,  # Contrast lower bound
        upper=1.2   # Contrast upper bound
    )
)


# Merge augmentations by applying train_datagen_extra preprocessing to train_datagen
def combined_preprocessing(img):
    # Apply train_datagen's augmentation (this is done directly when generating batches)
    img = train_datagen_extra.preprocessing_function(img)  # Apply extra transformations (e.g., brightness, saturation)
    return img

# Final train_datagen with all augmentations
train_datagen.preprocessing_function = combined_preprocessing

val_datagen = ImageDataGenerator(rescale=1.0/255.0)   # Rescale validation images (without augmentation)

# Split into training and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['encoded_label'], random_state=42)

# # Define the directory paths
train_dir = '/kaggle/input/bttai-ajl/train/train/'

def create_generator(dataframe, directory, batch_size=32, target_size=(224, 224), datagen=train_datagen):
    # Fill in the correct flow_from_dataframe parameters
    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col='file_path',  # Use combined path
        y_col='encoded_label',
        target_size=target_size, # Reshape images to 224x224
        batch_size=batch_size,
        class_mode='raw', # Outputs raw labels (for classification)
        validate_filenames=False  # Disable strict filename validation
    )
    return generator

# Create generators
train_generator = create_generator(train_data, train_dir) # Uses augmentation
val_generator = create_generator(val_data, train_dir, datagen=val_datagen) # No augmentation, only rescaling

"""## 4. Ensemble Testing"""

# 6. Make Predictions on Test Data
def preprocess_test_data(test_df, directory, target_size=(224, 224), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,
        x_col="md5hash",  # Use md5hash since test data doesn't have labels
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,  # No labels for test data
        shuffle=False  # Keep order for consistent predictions
    )

    return test_generator

# Load test data
test_dir = '/kaggle/input/bttai-ajl/test/test/'
test_generator = preprocess_test_data(test_df, test_dir)

"""### Soft Voting"""

# Load all trained models
model_paths = [
    "/kaggle/input/fine-tuned-models/EfficientNetB4_model_v2_100layers.keras",
    "/kaggle/input/fine-tuned-models/MobileNetV3Large_finetuned_v2_100_layers.keras",
    "/kaggle/input/fine-tuned-models/NASNetMobile_finetuned_model_v2_100layers.h5",
    "/kaggle/input/fine-tuned-models/densenet121_finetuned_model_v2_100unfrozen.h5",
    "/kaggle/input/fine-tuned-models/efficientnet_b3_finetuned-v2-100layers.keras",
    "/kaggle/input/fine-tuned-models/resnet50_model_v2_100layers.h5"
]

models = [load_model(path) for path in model_paths]

# Function to preprocess test data
def preprocess_test_data(test_df, directory, target_size=(224, 224), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,
        x_col="md5hash",
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    return test_generator

# Load test data
test_dir = '/kaggle/input/bttai-ajl/test/test/'
test_generator = preprocess_test_data(test_df, test_dir)

# Generate predictions from all models
all_predictions = [model.predict(test_generator) for model in models]

# Convert predictions to numpy arrays
all_predictions = np.array(all_predictions)  # Shape: (num_models, num_samples, num_classes)

# Use soft voting (average probabilities)
average_predictions = np.mean(all_predictions, axis=0)  # Shape: (num_samples, num_classes)

# Convert to class labels
predicted_labels = np.argmax(average_predictions, axis=1)

# Decode class labels back to original string labels
decoded_labels = label_encoder.inverse_transform(predicted_labels)

# Save predictions to a CSV file
submission_df = pd.DataFrame({
    "md5hash": test_df["md5hash"],
    "label": decoded_labels
})

# Remove '.jpg' from the 'md5hash' column
submission_df['md5hash'] = submission_df['md5hash'].str.replace('.jpg', '', regex=False)

# Save to CSV
submission_file = "submission_soft.csv"
submission_df.to_csv(submission_file, index=False)

print(f"Submission file saved as {submission_file}")

submission_df

"""### Hard Voting"""

# Load all trained models
model_paths = [
    "/kaggle/input/fine-tuned-models/EfficientNetB4_model_v2_100layers.keras",
    "/kaggle/input/fine-tuned-models/densenet121_finetuned_model_v2_100unfrozen.h5",
    "/kaggle/input/fine-tuned-models/efficientnet_b3_finetuned-v2-100layers.keras",
    "/kaggle/input/fine-tuned-models/resnet50_model_v2_100layers.h5"
]

models = [load_model(path) for path in model_paths]

# Function to preprocess test data
def preprocess_test_data(test_df, directory, target_size=(224, 224), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,
        x_col="md5hash",
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    return test_generator

# Load test data
test_dir = '/kaggle/input/bttai-ajl/test/test/'
test_generator = preprocess_test_data(test_df, test_dir)

# Generate predictions from all models
all_predictions = [model.predict(test_generator) for model in models]

# Convert predictions to numpy arrays
all_predictions = np.array(all_predictions)  # Shape: (num_models, num_samples, num_classes)

# Convert probabilities to class labels for each model
predicted_class_labels = np.argmax(all_predictions, axis=2)  # Shape: (num_models, num_samples)

# Perform hard voting (majority vote)
final_predictions, _ = mode(predicted_class_labels, axis=0)  # Shape: (1, num_samples)

# Reshape final predictions
final_predictions = final_predictions.flatten()  # Shape: (num_samples,)

# Decode class labels back to original string labels
decoded_labels = label_encoder.inverse_transform(final_predictions)

# Save predictions to a CSV file
submission_df = pd.DataFrame({
    "md5hash": test_df["md5hash"],
    "label": decoded_labels
})

# Remove '.jpg' from the 'md5hash' column
submission_df['md5hash'] = submission_df['md5hash'].str.replace('.jpg', '', regex=False)

# Save to CSV
submission_file = "submission_new.csv"
submission_df.to_csv(submission_file, index=False)

print(f"Submission file saved as {submission_file}")

submission_df