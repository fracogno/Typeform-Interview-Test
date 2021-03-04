import csv
import numpy as np
import pickle
import json
import string
from pathlib import Path


def get_base_path():
    return str(Path(__file__).parent.parent.parent) + "/"


def save_pickle(path, array):
    with open(path, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_json(path, array):
    with open(path, 'w') as f:
        json.dump(array, f)


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def import_csv_data(filename):
    headers = None
    X, Y = [], []
    with open(filename, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for row in reader:
            if headers is None:
                headers = {row[i].rstrip(): i for i in range(len(row))}
            else:
                X.append(row[3:])
                if float(row[1]) == 0.:
                    assert(float(row[2]) == 0.)
                    Y.append(0.)
                else:
                    Y.append(float(row[2]) / float(row[1]))
                    assert(Y[-1] <= 1.)

    return np.array(X).astype(np.float), np.array(Y).astype(np.float)


def import_csv_text(filename):
    X = []
    headers = None
    with open(filename, newline='\n') as file:
        reader = csv.reader(file, delimiter='\n', quotechar='|')
        for row in reader:
            if headers is None:
                headers = 1
            else:
                X.append(row[0])

    return X


def clean_strings(X):
    translator = str.maketrans('', '', string.punctuation)

    cleaned = []
    for sample in X:
        sample = sample.replace("<strong>", "").replace("</strong>", "").replace("<em>", "").replace("</em>", "").replace("<br />", "")
        cleaned.append(sample)
    return [sample.translate(translator).lower() for sample in cleaned]


def load_pretrained_embeddings_matrix(path, dictionary, embedding_dim):
    embeddings_index = {}

    with open(path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    # Prepare embedding matrix
    embedding_matrix = np.zeros((len(dictionary.index_to_word), embedding_dim))
    for word, i in dictionary.word_to_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
