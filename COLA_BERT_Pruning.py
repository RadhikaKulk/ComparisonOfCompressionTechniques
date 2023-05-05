!pip install -q tensorflow==2.6.0
!pip install -q transformers
!pip install -q datasets==1.12.1
!pip install -q sentencepiece
!pip install -U datasets

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef

# Custom MCC metric
class MCC(tf.keras.metrics.Metric):
    def __init__(self, name="mcc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.greater_equal(tf.nn.softmax(y_pred, axis=-1)[:, 1], 0.5)
        y_pred = tf.cast(y_pred, tf.bool)

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)), dtype=tf.float32)))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False)), dtype=tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)), dtype=tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False)), dtype=tf.float32)))

    def result(self):
        numerator = self.true_positives * self.true_negatives - self.false_positives * self.false_negatives
        denominator = tf.sqrt((self.true_positives + self.false_positives) * (self.true_positives + self.false_negatives) * (self.true_negatives + self.false_positives) * (self.true_negatives + self.false_negatives))
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_states(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Load the CoLA dataset
cola_dataset = load_dataset("glue", "cola")
train_dataset = cola_dataset['train']
val_dataset = cola_dataset['validation']
test_dataset = cola_dataset['test']

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
batches = 10

# Tokenize and prepare the dataset for training
def encode(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", return_tensors="tf")

train_encoded_dataset = train_dataset.map(encode, batched=True)
val_encoded_dataset = val_dataset.map(encode, batched=True)

# Convert the datasets to TensorFlow format
def to_tf_dataset(encoded_dataset):
    input_ids = encoded_dataset["input_ids"]
    attention_mask = encoded_dataset["attention_mask"]
    labels = encoded_dataset["label"]

    return tf.data.Dataset.from_tensor_slices(({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, labels))

train_tf_dataset = to_tf_dataset(train_encoded_dataset)
val_tf_dataset = to_tf_dataset(val_encoded_dataset)

train_tf_dataset = train_tf_dataset.shuffle(len(train_encoded_dataset)).batch(batches).repeat(-1)
val_tf_dataset = val_tf_dataset.batch(batches)

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

# Compile the model with the optimizer, loss function, and evaluation metric
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
mcc_metric = MCC()  # Use the custom MCC metric
model.compile(optimizer=optimizer, loss=loss, metrics=[mcc_metric])

# Train the model
model.fit(train_tf_dataset, epochs=20, steps_per_epoch=len(train_encoded_dataset)//batches, validation_data=val_tf_dataset)

def preprocess_data(texts, labels):
    input_features = tokenizer(texts, return_tensors="tf", padding=True, truncation=True)
    input_features["label"] = labels
    return input_features

val_features = preprocess_data(val_dataset["sentence"], val_dataset["label"])
# Convert the BatchEncoding object to a dictionary
val_features_dict = {
    "input_ids": val_features["input_ids"],
    "attention_mask": val_features["attention_mask"],
    "token_type_ids": val_features["token_type_ids"],
}


def prune_encoder_layer(layer, sparsity):
    weights = layer.get_weights()
    abs_weights = np.abs(weights[0])
    threshold = np.percentile(abs_weights, sparsity)
    mask = abs_weights >= threshold
    pruned_weights = weights[0] * mask.astype(np.float32)
    layer.set_weights([pruned_weights] + weights[1:])
    return layer

def prune_model(model, sparsity):
    pruned_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    
    for index, layer in enumerate(model.bert.encoder.layer):
        pruned_layer = prune_encoder_layer(layer, sparsity)
        pruned_model.bert.encoder.layer[index].set_weights(pruned_layer.get_weights())

    # Copy weights of the embeddings and the classifier layers
    pruned_model.bert.embeddings.set_weights(model.bert.embeddings.get_weights())
    pruned_model.classifier.set_weights(model.classifier.get_weights())

    return pruned_model

from transformers import BertTokenizer
from sklearn.metrics import matthews_corrcoef

# Prune the model with 50% sparsity
pruned_model = prune_model(model, 30)
pruned_model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Evaluate the pruned model and calculate MCC
y_true = np.array(val_dataset["label"])
y_pred = np.argmax(pruned_model.predict(val_features_dict).logits, axis=1)
mcc = matthews_corrcoef(y_true, y_pred)
print("Matthews Correlation Coefficient (MCC) for the pruned model:", mcc)

