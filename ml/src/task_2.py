import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

from models import task_2
from utils import misc, dictionary


def train(params):
    base_path = misc.get_base_path()
    data_path = "dataset/form_questions.csv"

    # Import data
    X = misc.import_csv_text(base_path + data_path)

    # Data preprocessing
    X = misc.clean_strings(X)
    print(len(X))
    X_tokens = [word_tokenize(sample) for sample in X]
    max_length_sentence = max([len(sample) for sample in X_tokens])

    # Create dictionary and map words to integers
    dict_tracked = dictionary.LanguageDictionary(X_tokens, max_length_sentence)
    X_mapped = np.array([np.array(dict_tracked.text_to_indices(["<START>"] + sample + ["<END>"])) for sample in X_tokens])

    # Pad sequences
    X_mapped = tf.keras.preprocessing.sequence.pad_sequences(X_mapped, maxlen=max_length_sentence, dtype='int32', padding='post')
    #print(dict_tracked.indices_to_text(X_mapped[10]))

    # Split dataset and create TF datasets
    X_train, X_test = train_test_split(X_mapped, test_size=0.15, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000).batch(params["batch_size"])
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test).shuffle(50).batch(params["batch_size"])

    # Load pretrained embeddings (from pickles to save time)
    if os.path.isfile(base_path + "pickles/task_1_embeddings_matrix"):
        embeddings_matrix = misc.load_pickle(base_path + "pickles/task_1_embeddings_matrix")
    else:
        embeddings_matrix = misc.load_pretrained_embeddings_matrix(base_path + "dataset/glove.6B/glove.6B." + str(params["embedding_size"])+"d.txt",
                                                                   dict_tracked, params["embedding_size"])
        misc.save_pickle(base_path + "pickles/task_1_embeddings_matrix", embeddings_matrix)
    encoder = task_2.Encoder(embeddings_matrix, len(dict_tracked.index_to_word), params["embedding_size"], 128, params["batch_size"])

    # Training
    for epoch in range(params["epochs"]):

        enc_hidden = encoder.initialize_hidden_state()

        for batch in train_dataset:
            print(batch.shape)
            output, state = encoder(batch, enc_hidden)
            print(output.shape)
            print(state.shape)


            break


if __name__ == "__main__":
    train({"batch_size": 64, "epochs": 1, "embedding_size": 100})
