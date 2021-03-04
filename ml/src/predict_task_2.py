import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
from sklearn.decomposition import PCA

from models import task_2
from utils import misc


def predict():
    base_path = misc.get_base_path()

    X_mapped = misc.load_pickle(base_path + "pickles/X_mapped")
    dict_tracked = misc.load_pickle(base_path + "pickles/dictionary")
    params = misc.load_pickle(base_path + "pickles/params")
    embeddings_matrix = misc.load_pickle(base_path + "pickles/task_2_embeddings_matrix")
    vocab_size = len(dict_tracked.index_to_word)

    # Encoder only needed
    encoder = task_2.Encoder(embeddings_matrix, vocab_size, params["embedding_size"], params["units"], params["batch_size"])
    encoder.load_weights(base_path + "checkpoints/task_2/ckpt-encoder")

    # Extract features
    dataset = tf.data.Dataset.from_tensor_slices(X_mapped).batch(1)
    sentences = []
    features = []
    for batch in dataset:
        batch = batch.numpy()
        batch = np.array([[sample for sample in batch[0] if sample != 0. and sample != 1 and sample != 2]])
        hidden = [tf.zeros((1, params["units"]))]
        _, state = encoder(batch, hidden)
        sentences.append(dict_tracked.indices_to_text(batch[0]))
        features.append(state[0])

    features = np.array(features)
    print(features.shape)

    # Apply PCA to features
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features)
    print("Shape after PCA :", str(principal_components.shape))

    # Save labels separately line-by-line
    log_dir = base_path + "checkpoints/task_2"
    with open(log_dir + "/metadata.tsv", "w") as f:
        for label in sentences:
            f.write("{}\n".format(label))

    weights = tf.Variable(principal_components)
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(log_dir + "/embedding.ckpt")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


if __name__ == "__main__":
    predict()
