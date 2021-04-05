import argparse
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import kerastuner as kt
from kerastuner.tuners import Hyperband
from kapre.composed import get_melspectrogram_layer
import kapre
import tensorflow as tf
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow import keras

from transformer_encoder import Encoder
from train import DataGenerator
from visual_transformer import mlp, Patches, PatchEncoder

src_root = 'clean'
batch_size = 16
dt = 1.0
sr = 16000
N_CLASSES = 10

def build_vit(hp):

    dropout = hp.Float('dropout', 0.05, 0.4, sampling='log')
    image_size = 128  # We'll resize input images to this size
    patch_size = hp.Int('transformer_layers', 4, 8, step=2)  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = hp.Int('projection_dim', 8, 40, step=8)
    num_heads = hp.Int('attention_heads', 2, 8, step=2)
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = hp.Int('transformer_layers', 2, 6, step=1)
    mlp_head_units = hp.Int('mlp_head_units', 10, 20, step=2)

    input_shape = (int(sr*dt), 1)

    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=sr,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)  

    # Augment data.
    # augmented = data_augmentation(inputs)

    resize = layers.experimental.preprocessing.Resizing(image_size, image_size)(x)

    # Create patches.
    patches = Patches(patch_size)(resize)

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #Add Pooling for dimensionality reduction
    representation = tf.keras.layers.Reshape(target_shape=tf.expand_dims(representation, axis=-1).shape[1:])(representation)
    pooling = layers.MaxPool2D()(representation)

    representation = layers.Flatten()(pooling)

    # Add MLP.
    features = mlp(representation, hidden_units=[mlp_head_units], dropout_rate=dropout)

    # Classify outputs.
    logits = layers.Dense(N_CLASSES)(features)
    # Create the Keras model.
    model = keras.Model(inputs=i.input, outputs=logits)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model    

def optimize_model():

    params = {'N_CLASSES':len(os.listdir(src_root)),
              'SR':sr,
              'DT':dt}

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.15,
                                                                  random_state=0)

    assert len(label_train) >= batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(len(set(label_val)), params['N_CLASSES']))

    tg = DataGenerator(wav_train, label_train, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)


    tuner = kt.Hyperband(build_vit,
                        objective='val_accuracy',
                        max_epochs=80,
                        hyperband_iterations=2)

    tuner.search(tg,
                validation_data=vg,
                epochs=50,
                callbacks=[], 
                workers=16)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameters)


if __name__ == '__main__':
    optimize_model()

