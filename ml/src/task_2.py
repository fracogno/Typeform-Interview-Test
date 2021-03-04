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
    X_tokens = [word_tokenize(sample) for sample in X if sample != ""]
    X_tokens = [sample for sample in X_tokens if len(sample) <= 50]
    max_length_sentence = max([len(sample) for sample in X_tokens])

    # Create dictionary and map words to integers
    dict_tracked = dictionary.LanguageDictionary(X_tokens, max_length_sentence)
    vocab_size = len(dict_tracked.index_to_word)
    X_mapped = np.array([np.array(dict_tracked.text_to_indices(["<START>"] + sample + ["<END>"])) for sample in X_tokens])

    # Pad sequences
    X_mapped = tf.keras.preprocessing.sequence.pad_sequences(X_mapped, maxlen=max_length_sentence, dtype='int32', padding='post')
    misc.save_pickle(base_path + "pickles/X_mapped", X_mapped)
    #print(dict_tracked.indices_to_text(X_mapped[10]))

    # Split dataset and create TF datasets
    X_train, X_test = train_test_split(X_mapped, test_size=0.15, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000).batch(params["batch_size"])
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test).shuffle(50).batch(params["batch_size"])

    # Load pretrained embeddings (from pickles to save time)
    if os.path.isfile(base_path + "pickles/task_2_embeddings_matrix"):
        embeddings_matrix = misc.load_pickle(base_path + "pickles/task_2_embeddings_matrix")
    else:
        embeddings_matrix = misc.load_pretrained_embeddings_matrix(base_path + "dataset/glove.6B/glove.6B." + str(params["embedding_size"])+"d.txt",
                                                                   dict_tracked, params["embedding_size"])
        misc.save_pickle(base_path + "pickles/task_2_embeddings_matrix", embeddings_matrix)

    # Models
    encoder = task_2.Encoder(embeddings_matrix, vocab_size, params["embedding_size"], params["units"], params["batch_size"])
    decoder = task_2.Decoder(vocab_size, encoder.embedding, params["units"])

    # Optimizers and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Save useful pickles
    misc.save_pickle(base_path + "pickles/dictionary", dict_tracked)
    misc.save_pickle(base_path + "pickles/X_mapped", X_mapped)
    misc.save_pickle(base_path + "pickles/params", params)

    # Training
    best_loss = 10000.
    for epoch in range(params["epochs"]):

        enc_hidden = encoder.initialize_hidden_state()

        total_loss = 0.
        steps = 0.
        for batch in train_dataset:
            if len(batch) != params["batch_size"]:
                continue
            dec_input = tf.expand_dims([dict_tracked.word_to_index['<START>']] * params["batch_size"], 1)
            batch_loss = task_2.train_step(batch, batch, encoder, decoder, enc_hidden, optimizer, dec_input, loss_object, False)
            total_loss += batch_loss
            steps += 1
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps))

        total_loss = 0.
        steps = 0.
        for batch in test_dataset:
            if len(batch) != params["batch_size"]:
                continue
            dec_input = tf.expand_dims([dict_tracked.word_to_index['<START>']] * params["batch_size"], 1)
            batch_loss = task_2.train_step(batch, batch, encoder, decoder, enc_hidden, optimizer, dec_input, loss_object, True)
            total_loss += batch_loss
            steps += 1
        print('Validation epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps))

        if total_loss < best_loss:
            encoder.save_weights(base_path + "checkpoints/task_2/ckpt-encoder")
            best_loss = total_loss


if __name__ == "__main__":
    train({"batch_size": 32, "epochs": 200, "embedding_size": 100, "units": 32})
