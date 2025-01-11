import os
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Preprocessing for Training and Validation Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Split training into training and validation subsets
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Preprocessing for Testing Data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Verify Class Mapping
class_indices = train_generator.class_indices
print("Class Mapping:", class_indices)