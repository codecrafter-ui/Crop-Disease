import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import ResNet50, DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
import os

IMG_SIZE = 224
BATCH_SIZE = 65
EPOCHS = 30

train_dir = os.path.join("..", "data", "train")
model_save_dir = os.path.join("..", "models")
os.makedirs(model_save_dir, exist_ok=True)

num_classes = len(os.listdir(train_dir))
print(f"Detected {num_classes} classes for training.")

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.30
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

def build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
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
    base = base_model(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=predictions)
    return model

models_to_train = {
    "CNN": build_cnn_model,
    "ResNet50": lambda: build_transfer_model(ResNet50, "ResNet50"),
    "DenseNet121": lambda: build_transfer_model(DenseNet121, "DenseNet121"),
}

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

for name, model_builder in models_to_train.items():
    print(f"\n--- Training {name} ---")
    model = model_builder()

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_filepath = os.path.join(model_save_dir, f'{name}_best.h5')
    checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss')

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    model.save(os.path.join(model_save_dir, f'{name}_final.h5'))
    print(f"{name} training complete. Model saved to {checkpoint_filepath}")
