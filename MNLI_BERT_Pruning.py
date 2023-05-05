!pip install -q tensorflow==2.6.0
!pip install -q transformers
!pip install -q datasets==1.12.1
!pip install -q sentencepiece
!pip install -U datasets
!pip install -q tensorflow-model-optimization

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load the MNLI dataset
mnli_dataset = load_dataset("glue", "mnli")
train_dataset = mnli_dataset['train']
val_dataset = mnli_dataset['validation_matched']
test_dataset = mnli_dataset['test_matched']

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
batches = 10

# Tokenize and prepare the dataset for training
def encode(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", return_tensors="tf")

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
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_tf_dataset, epochs=10, steps_per_epoch=len(train_encoded_dataset)//batches, validation_data=val_tf_dataset)


# Preprocess the validation data
def preprocess_data(premises, hypotheses, labels):
    input_features = tokenizer(premises, hypotheses, return_tensors="tf", padding=True, truncation=True)
    input_features["label"] = labels
    return input_features

val_features = preprocess_data(val_dataset["premise"], val_dataset["hypothesis"], val_dataset["label"])

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

    pruned_model.bert.embeddings.set_weights(model.bert.embeddings.get_weights())
    pruned_model.classifier.set_weights(model.classifier.get_weights())

    return pruned_model

# Prune the model with 30% sparsity
pruned_model = prune_model(model, 30)
pruned_model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Evaluate the pruned model and calculate the accuracy
y_true = np.array(val_dataset["label"])
y_pred = np.argmax(pruned_model.predict(val_features_dict).logits, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy for the pruned model:", accuracy)