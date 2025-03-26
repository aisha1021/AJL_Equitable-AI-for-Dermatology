"""
## 1. Import Necessary Libraries
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from collections import Counter
import matplotlib.pyplot as plt
import pickle

# Explanation:
# - pandas and numpy: for data manipulation
# - sklearn: for splitting data and encoding labels
# - tensorflow.keras: for building and training the neural network

"""## 2. Load Data

Make sure to verify the file paths if you're running on a different platform.
"""

# 2. Load Data
train_df = pd.read_csv('/kaggle/input/bttai-ajl-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/bttai-ajl-2025/test.csv')

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

# 4. Determine Maximum Samples per Class & Fitzpatrick Scale
max_samples_per_class = train_df['label'].value_counts().max()
max_samples_per_scale = train_df['fitzpatrick_scale'].value_counts().max()

# 5. Perform Data Augmentation to Balance Classes & Fitzpatrick Scale
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
train_dir = '/kaggle/input/bttai-ajl-2025/train/train/'

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

"""## 4. Build the model

"""

# Define input shape for MobileNetV3-Large
input_shape = (224, 224, 3)  # MobileNetV3-Large uses 224x224 input size
input_layer = Input(shape=input_shape, name="input_layer")

weights='/kaggle/input/mobilenet-v3-large/weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5'

# Load MobileNetV3-Large without top layers
base_model = MobileNetV3Large(weights=weights, include_top=False, input_tensor=input_layer)

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.4)(x)
output_layer = Dense(21, activation='softmax')(x)  # Adjust number of classes as needed

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print Model Summary
model.summary()

"""## 5. Train the Model

"""

# 11. Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=8,
    callbacks=[early_stopping, reduce_lr]
)

# Save the entire model, including weights, architecture, and optimizer state
model.save('MobileNetV3Large_model.h5')

# Save history
with open('MobileNetV3Large-history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Load history
with open('MobileNetV3Large-history.pkl', 'rb') as f:
    history = pickle.load(f)

acc = history['accuracy']
val_acc = history['val_accuracy']

loss = history['loss']
val_loss = history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""## 6. Fine-tune the Model

"""

# **Fine-Tuning Stage (Unfreeze Some Layers)**

# Load the pre-trained model (with frozen layers)
model = tf.keras.models.load_model('/kaggle/input/mobilenet-v3-model/mobilenet_v3_model.keras')

# Unfreeze last 60 layers of the loaded model
for layer in model.layers[-30:]:
    layer.trainable = True

# Ensure BatchNorm Layers Are Trainable
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Compile Model Again (Lower Learning Rate)
model.compile(optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train Again for Fine-Tuning
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=4,  # Fine-tuning for a few more epochs
    callbacks=[early_stopping, reduce_lr]
)

# Save history
with open('MobileNetV3Large-finetuned-history.pkl', 'wb') as f:
    pickle.dump(history_fine.history, f)

# Load history
with open('MobileNetV3Large-finetuned-history.pkl', 'rb') as f:
    history_fine = pickle.load(f)

acc += history_fine['accuracy']
val_acc += history_fine['val_accuracy']

loss += history_fine['loss']
val_loss += history_fine['val_loss']

# Define initial_epochs (number of epochs before fine-tuning starts)
initial_epochs = 8

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

model.save('mobilenet_v3_finetuned_model.keras')

"""## 7. Make Predictions on Test Data"""

loaded_model = tf.keras.models.load_model('/kaggle/working/mobilenet_v3_finetuned_model.keras')

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
test_dir = '/kaggle/input/bttai-ajl-2025/test/test/'
test_generator = preprocess_test_data(test_df, test_dir)

"""## 8. Generate Predictions"""

# Generate predictions based on the trained model
# Then, save the predictions into a CSV file for submission

# Load test data
test_dir = '/kaggle/input/bttai-ajl-2025/test/test/'
test_generator = preprocess_test_data(test_df, test_dir)

# Generate predictions
predictions = loaded_model.predict(test_generator) # change this after all three models have run

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Decode the predicted labels back to their original string labels
decoded_labels = label_encoder.inverse_transform(predicted_labels)

# Save predictions to a CSV file for submission
submission_df = pd.DataFrame({
    "md5hash": test_df["md5hash"],
    "label": decoded_labels
})

# Remove '.jpg' from the 'md5hash' column
submission_df['md5hash'] = submission_df['md5hash'].str.replace('.jpg', '', regex=False)

# Save to CSV
submission_file = "submission_efficientnetb4.csv"
submission_df.to_csv(submission_file, index=False)

print(f"Submission file saved as {submission_file}")

submission_df

"""## 9. Evaluate"""

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")