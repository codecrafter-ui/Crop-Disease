import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping 
from keras.applications import ResNet50, DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
import os

# --- Configuration Updated ---
IMG_SIZE = 224
BATCH_SIZE = 99
# Increased epochs to 100 as per teacher's requirement
EPOCHS = 100 
train_dir = os.path.join("..", "data", "train")
model_save_dir = os.path.join("..", "models")
os.makedirs(model_save_dir, exist_ok=True)

num_classes = len(os.listdir(train_dir))
print(f"Detected {num_classes} classes for training.")

# --- Data Generators ---
train_gen = ImageDataGenerator(
    rescale=1./255, 
    # CHANGED: validation_split is now 0.20 (20%) for an 80/20 split
    validation_split=0.20,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- Model Building Functions ---

def build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
    """Builds a simple Convolutional Neural Network model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_transfer_model(base_model, model_name):
    """Builds a fine-tuned model using a pre-trained base (ResNet/DenseNet)."""
    base = base_model(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False 

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=predictions)
    return model

# --- Training Pipeline ---

models_to_train = {
    "CNN": build_cnn_model,
    "ResNet50": lambda: build_transfer_model(ResNet50, "ResNet50"),
    "DenseNet121": lambda: build_transfer_model(DenseNet121, "DenseNet121"),
}

# Define Early Stopping outside the loop to monitor validation loss and stop if no improvement for 10 epochs
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

for name, model_builder in models_to_train.items():
    print(f"\n--- Training {name} ---")
    model = model_builder()
    
    # Use keras.optimizers
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    checkpoint_filepath = os.path.join(model_save_dir, f'{name}_best.h5')
    # Checkpoint still saves the best model based on val_loss
    checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss')

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        # Added EarlyStopping to the callbacks list
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model 
    model.save(os.path.join(model_save_dir, f'{name}_final.h5'))
    print(f"{name} training complete. Model saved to {checkpoint_filepath}")