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

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

best_model_path = '/content/drive/MyDrive/Colab Notebooks/Resnet model/resetcifar_best_model.h5'
best_model = tf.keras.models.load_model(best_model_path)

# Pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
}
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model = prune_low_magnitude(best_model, **pruning_params)

# Fine-tune the pruned model
opt = Adam(learning_rate=1e-5)
pruned_model.compile(
    loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
pruned_model.fit(train_images, train_labels, batch_size=128, epochs=10,
                 validation_split=0.1, callbacks=callbacks)


# Strip pruning wrappers
stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# PQAT
quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
    stripped_pruned_model)
pqat_model = tfmot.quantization.keras.quantize_apply(
    quant_aware_annotate_model,
    tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())

pqat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
pqat_model.fit(train_images, train_labels, batch_size=128, epochs=10, validation_split=0.1)