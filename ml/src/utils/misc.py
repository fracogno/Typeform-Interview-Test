import csv
import numpy as np
import pickle
import json
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


def import_csv_data(filename, has_headers):
    headers = None
    X, Y = [], []
    with open(filename, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for row in reader:
            if has_headers and headers is None:
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
