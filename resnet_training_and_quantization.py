!pip install -U -q tensorflow-model-optimization numpy

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import math

# Load and preprocess the dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Load and adapt a pre-trained ResNet50 model for the CIFAR-10 dataset
base_model = applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(32, 32, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_images)

# Cosine annealing learning rate schedule
def cosine_annealing_schedule(epoch, lr):
    T_max = 100
    eta_min = 1e-6
    return eta_min + (0.5 * (lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)))

lr_scheduler = LearningRateScheduler(cosine_annealing_schedule)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

# Fine-tuning: Unfreeze the top layers
for layer in base_model.layers[-5:]:
    layer.trainable = True

# Compile the model with a smaller learning rate
opt = Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Save the best model
model_checkpoint = ModelCheckpoint(filepath='/content/drive/MyDrive/Colab Notebooks/Resnet model/resetcifar_best_model.h5',
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   mode='max')

# Train the model
model.fit(datagen.flow(train_images, train_labels, batch_size=128),
          steps_per_epoch=len(train_images) / 128,
          epochs=10,
          validation_data=(val_images, val_labels),
          callbacks=[lr_scheduler, early_stopping, model_checkpoint])


# Quantize the model
best_model_path = '/content/drive/MyDrive/Colab Notebooks/Resnet model/resetcifar_best_model.h5'
best_model = tf.keras.models.load_model(best_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_model)

# Load the quantized model
interpreter = tf.lite.Interpreter(model_content=quantized_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to evaluate the quantized model
def evaluate_quantized_model(interpreter, test_images, test_labels):
    total_seen = 0
    num_correct = 0
    
    for img, label in zip(test_images, test_labels):
        inp = img.reshape((1, 32, 32, 3))
        total_seen += 1
        interpreter.set_tensor(input_details[0]['index'], inp.astype('float32'))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(predictions) == label[0]:
            num_correct += 1

    return float(num_correct) / float(total_seen)

# Evaluate the quantized model
quantized_acc = evaluate_quantized_model(interpreter, test_images, test_labels)
print("Quantized model accuracy: {:.2f}%".format(quantized_acc * 100))

