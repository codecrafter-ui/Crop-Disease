import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = 224
test_dir = os.path.join("..", "data", "test")
model_name = input("Enter the model name (e.g., CNN, ResNet50, DenseNet121): ")
model_path = os.path.join("..", "models", f"{model_name}_best.h5")
figure_dir = os.path.join("..", "figures")
os.makedirs(figure_dir, exist_ok=True)

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_data.class_indices.keys())

try:
    model = keras.models.load_model(model_path)
    print(f"Successfully loaded model: {model_name}")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    exit()

predictions = model.predict(test_data)
y_true = test_data.classes
y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"\n--- Performance Metrics for {model_name} (Test Set) ---")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix ({model_name})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
cm_filepath = os.path.join(figure_dir, f'cm_{model_name}.png')
plt.savefig(cm_filepath)
print(f"Confusion Matrix saved to {cm_filepath}")
