import os
import csv
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me
import plot_results as pr

from train_generative import build_generative_model
from sklearn.linear_model import LogisticRegression

# Directory where trained model will be saved
TRAIN_DIR = "./trained"

def preprocess_sentence(text, front_pad='\n ', end_pad=''):
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    return text

def encode_sentence(model, text, char2idx, layer_idx):
    text = preprocess_sentence(text)

    # Reset LSTMs hidden and cell states
    model.reset_states()

    for c in text.split(" "):
        # Add the batch dimension
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)
            predictions = model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", c)

    h_state, c_state = model.get_layer(index=layer_idx).states

    # remove the batch dimension
    #h_state = tf.squeeze(h_state, 0)
    c_state = tf.squeeze(c_state, 0)

    return tf.math.tanh(c_state).numpy()

def build_dataset(datapath, generative_model, char2idx, layer_idx):
    xs, ys = [], []

    csv_file = open(datapath, "r")
    data = csv.DictReader(csv_file)

    for row in data:
        label = int(row["label"])
        filepath = row["filepath"]

        data_dir = os.path.dirname(datapath)
        phrase_path = os.path.join(data_dir, filepath) + ".mid"
        encoded_path = os.path.join(data_dir, filepath) + ".npy"

        # Load midi file as text
        if os.path.isfile(encoded_path):
            encoding = np.load(encoded_path)
        else:
            text, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)

            # Encode midi text using generative lstm
            encoding = encode_sentence(generative_model, text, char2idx, layer_idx)

            # Save encoding in file to make it faster to load next time
            np.save(encoded_path, encoding)

        xs.append(encoding)
        ys.append(label)

    return np.array(xs), np.array(ys)

def train_classifier_model(train_dataset, test_dataset, C=2**np.arange(-8, 1).astype(float), seed=42, penalty="l1"):
    trX, trY = train_dataset
    teX, teY = test_dataset

    scores = []

    # Hyper-parameter optimization
    for i, c in enumerate(C):
        logreg_model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i, solver="liblinear")
        logreg_model.fit(trX, trY)

        score = logreg_model.score(teX, teY)
        scores.append(score)

    c = C[np.argmax(scores)]

    sent_classfier = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C), solver="liblinear")
    sent_classfier.fit(trX, trY)

    score =  sent_classfier.score(teX, teY) * 100.

    # Persist sentiment classifier
    with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
        pickle.dump(sent_classfier, f)

    # Get activated neurons
    sentneuron_ixs = get_activated_neurons(sent_classfier)

    # Plot results
    pr.plot_weight_contribs(sent_classfier.coef_)
    pr.plot_logits(trX, trY, sentneuron_ixs)

    return sentneuron_ixs, score

def get_activated_neurons(sent_classfier):
    neurons_not_zero = len(np.argwhere(sent_classfier.coef_))

    weights = sent_classfier.coef_.T
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

    if neurons_not_zero == 1:
        neuron_ixs = np.array([np.argmax(weight_penalties)])
    elif neurons_not_zero >= np.log(len(weight_penalties)):
        neuron_ixs = np.argsort(weight_penalties)[-neurons_not_zero:][::-1]
    else:
        neuron_ixs = np.argpartition(weight_penalties, -neurons_not_zero)[-neurons_not_zero:]
        neuron_ixs = (neuron_ixs[np.argsort(weight_penalties[neuron_ixs])])[::-1]

    return neuron_ixs

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_classifier.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test' , type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--cellix', type=int, required=True, help="LSTM layer to use as encoder.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild generative model from checkpoint
    generative_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    generative_model.load_weights(tf.train.latest_checkpoint(opt.model))
    generative_model.build(tf.TensorShape([1, None]))

    # Build dataset from encoded labelled midis
    train_dataset = build_dataset(opt.train, generative_model, char2idx, opt.cellix)
    test_dataset = build_dataset(opt.test, generative_model, char2idx, opt.cellix)

    # Train model
    sentneuron_ixs, score = train_classifier_model(train_dataset, test_dataset)

    print("Total Neurons Used:", len(sentneuron_ixs), "\n", sentneuron_ixs)
    print("Test Accuracy:", score)
