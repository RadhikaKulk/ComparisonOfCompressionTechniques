!pip install -q tensorflow-model-optimization
!pip install numpy
!pip install -U numpy
!pip install -U opencv-python

import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split
import math
import pathlib
from pathlib import Path

best_model_path = '/content/drive/MyDrive/Colab Notebooks/Resnet model/resetcifar_best_model.h5'
best_model = tf.keras.models.load_model(best_model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
base_tflite_model = converter.convert()

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

model = best_model

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.01,  # Decrease initial sparsity
                                                               final_sparsity=0.30,    
                                                               begin_step=0,
                                                               end_step=end_step),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

pruned_model = prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Use a smaller learning rate
pruned_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

fine_tuning_epochs = 20  # Increase the number of fine-tuning epochs
fine_tuning_batch_size = 128

pruned_model.fit(train_images, train_labels, batch_size=fine_tuning_batch_size, epochs=fine_tuning_epochs, validation_split=validation_split,
                  callbacks=callbacks)

_, pruned_model_accuracy = pruned_model.evaluate(test_images, test_labels, verbose=0)

pruned_model_3x = tfmot.sparsity.keras.strip_pruning(pruned_model)
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model_3x)
pruned_tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("/content/drive/MyDrive/Colab Notebooks/Resnet model")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
pruned_tflite_model_file = tflite_models_dir/"resnetcifar_pruned.tflite"
pruned_tflite_model_file.write_bytes(pruned_tflite_model)

# accuracy for tflite model

import time

interpreter_tflite = tf.lite.Interpreter(model_path=str(pruned_tflite_model_file))
interpreter_tflite.allocate_tensors()

def preprocess_input_data(images):
    input_data = np.expand_dims(images, axis=0)
    input_data = np.float32(input_data)
    return input_data

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

start_time = time.time()

correct_predictions = 0
total_predictions = len(test_images)

for i in range(total_predictions):
    input_data = preprocess_input_data(test_images[i])
    output_data = run_inference(interpreter_tflite, input_data)
    
    predicted_label = np.argmax(output_data)
    true_label = np.argmax(test_labels[i])
    
    if predicted_label == true_label:
        correct_predictions += 1

# Calculate the accuracy
accuracy = correct_predictions / total_predictions
end_time = time.time()
print("pruned tflite model accuracy: {:.2f}%".format(accuracy * 100))

duration = end_time - start_time
hours = duration // 3600
minutes = (duration - (hours * 3600)) // 60
seconds = duration - ((hours * 3600) + (minutes * 60))
inference_msg = f'pruned TFLite inference time: {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
print(inference_msg)

import tempfile
import os
def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped baseline model: %.2f bytes" % (get_gzipped_model_size(best_model)))
print("Size of gzipped baseline TFLite model: %.2f bytes" % (get_gzipped_model_size(base_tflite_model_file)))
print("Size of gzipped pruned TFLite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_model_file)))


