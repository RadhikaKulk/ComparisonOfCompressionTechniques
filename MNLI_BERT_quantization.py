!pip install -U --user datasets transformers tf-models-official==2.10.0 "tensorflow-text==2.10.*"
import os
import shutil

import tensorflow as tf
from transformers import TFBertMainLayer, BertConfig
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from official.nlp import optimization  # to create AdamW optimizer
import tensorflow_model_optimization as tfmot

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

tfds_name = 'glue/mnli'

bert_model_name = 'bert_en_uncased_L-12_H-768_A-12' 

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    }

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    }

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

def make_bert_preprocess_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

    Args:
    sentence_features: a list with the names of string-valued features.
    seq_length: an integer that defines the sequence length of BERT inputs.

    Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
    """

    input_segments = [tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft) for ft in sentence_features]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(tfhub_handle_preprocess)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]
    truncated_segments = segments
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                          arguments=dict(seq_length=seq_length),
                          name='packer')
    model_inputs = packer(truncated_segments)
    return tf.keras.Model(input_segments, model_inputs)


def build_classifier_model(num_classes):
    class Classifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.dense = tf.keras.layers.Dense(num_classes)

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense(x)
            return x
    model = Classifier(num_classes)
    return model


config = BertConfig.from_pretrained("bert-base-uncased")
bert = TFBertMainLayer(config)
inputs = tf.keras.layers.Input(shape=(None,), name='input_embeds', dtype='int32')
seq_emb = bert(inputs)[0]
last_token_emb = seq_emb[:, -1, :]
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(last_token_emb)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# Datap processing
tfds_info = tfds.builder(tfds_name).info

sentence_features = list(tfds_info.features.keys())
sentence_features.remove('idx')
sentence_features.remove('label')

available_splits = list(tfds_info.splits.keys())
train_split = 'train'
validation_split = 'validation'
test_split = 'test'
if tfds_name == 'glue/mnli':
    validation_split = 'validation_matched'
    test_split = 'test_matched'

num_classes = tfds_info.features['label'].num_classes
num_examples = tfds_info.splits.total_num_examples

print(f'Using {tfds_name} from TFDS')
print(f'This dataset has {num_examples} examples')
print(f'Number of classes: {num_classes}')
print(f'Features {sentence_features}')
print(f'Splits {available_splits}')

with tf.device('/job:localhost'):
  # batch_size=-1 is a way to load the dataset into memory
  in_memory_ds = tfds.load(tfds_name, batch_size=-1, shuffle_files=True)

# The code below is just to show some samples from the selected dataset
print(f'Here are some sample rows from {tfds_name} dataset')
sample_dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[train_split])

labels_names = tfds_info.features['label'].names
print(labels_names)
print()

sample_i = 1
for sample_row in sample_dataset.take(5):
    samples = [sample_row[feature] for feature in sentence_features]
    print(f'sample row {sample_i}')
    for sample in samples:
        print(sample.numpy())
    sample_label = sample_row['label']

    print(f'label: {sample_label} ({labels_names[sample_label]})')
    print()
    sample_i += 1

def get_configuration(glue_task):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if glue_task == 'glue/cola':
        metrics = "accuracy" #tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
    else:
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

    return metrics, loss

AUTOTUNE = tf.data.AUTOTUNE


def load_dataset_from_tfds(in_memory_ds, info, split, batch_size,
                           bert_preprocess_model):
    is_training = split.startswith('train')
    dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[split])
    num_examples = info.splits[split].num_examples

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, num_examples

epochs = 10
batch_size = 32
init_lr = 2e-5

print(f'Fine tuning {tfhub_handle_encoder} model')
bert_preprocess_model = make_bert_preprocess_model(sentence_features)

with strategy.scope():

    # metric have to be created inside the strategy scope
    metrics, loss = get_configuration(tfds_name)

    train_dataset, train_data_size = load_dataset_from_tfds(in_memory_ds, tfds_info, train_split, batch_size, bert_preprocess_model)
    steps_per_epoch = train_data_size // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = num_train_steps // 10

    validation_dataset, validation_data_size = load_dataset_from_tfds(in_memory_ds, tfds_info, validation_split, batch_size,bert_preprocess_model)
    validation_steps = validation_data_size // batch_size

    classifier_model = build_classifier_model(num_classes)
    optimizer = optimization.create_optimizer(
      init_lr=init_lr,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      optimizer_type='adamw')
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    classifier_model.fit(
      x=train_dataset,
      validation_data=validation_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_steps=validation_steps)
    classifier_model.save(tfds_name )

# quantization
from sklearn.metrics import accuracy_score


# Create a function to evaluate the model
def evaluate_model(interpreter, dataset):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    shape1 = list(input_details[0]["shape"]) 
    shape1[0] = batch_size
    shape2 = list(input_details[1]["shape"]) 
    shape2[0] = batch_size
    shape3 = list(input_details[2]["shape"]) 
    shape3[0] = batch_size
    interpreter.resize_tensor_input(input_details[0]['index'], shape1)
    interpreter.resize_tensor_input(input_details[1]['index'], shape2)
    interpreter.resize_tensor_input(input_details[2]['index'], shape3)
    interpreter.allocate_tensors()
    true_labels = []
    predicted_labels = []
       
    for inputs, labels in dataset.take(10):
        interpreter.set_tensor(input_details[0]["index"], inputs["input_word_ids"].numpy().astype(input_details[0]["dtype"]))
        interpreter.set_tensor(input_details[2]["index"], inputs["input_mask"].numpy().astype(input_details[2]["dtype"]))
        interpreter.set_tensor(input_details[1]["index"], inputs["input_type_ids"].numpy().astype(input_details[1]["dtype"]))
        interpreter.invoke()

        logits = interpreter.get_tensor(output_details[0]["index"])
        predictions = tf.argmax(logits, axis=1)
       
        true_labels.extend(labels.numpy().tolist())
        predicted_labels.extend(predictions.numpy().tolist())

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

batch_size=32
train_dataset, train_data_size = load_dataset_from_tfds(in_memory_ds, tfds_info, train_split, batch_size, bert_preprocess_model)
steps_per_epoch = train_data_size // batch_size
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = num_train_steps // 10

validation_dataset, validation_data_size = load_dataset_from_tfds(in_memory_ds, tfds_info, validation_split, batch_size,bert_preprocess_model)
validation_steps = validation_data_size // batch_size

# FP16 quantization

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(32, 128)
        yield {
            "input_word_ids": data.astype(np.int32),
            "input_type_ids": data.astype(np.int32),
            "input_mask":data.astype(np.int32)
            
        }
        
converter = tf.lite.TFLiteConverter.from_saved_model("mnli")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

model_out = "mnli_tflite_fp16"
with open(model_out, "wb") as f:
    f.write(tflite_quant_model)

# Evaluate the quantized model
interpreter = tf.lite.Interpreter(model_path=model_out, num_threads=8)

accuracy = evaluate_model(interpreter, validation_dataset)
print("Accuracy on mnli dataset:", accuracy)

