import tensorflow as tf
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from utils import misc, metrics
from models import task_1

# Paths
base_path = misc.get_base_path()
data_path = "dataset/completion_rate.csv"

# Load CSV for first time or load prepared pickle (saves time when developing)
if os.path.isfile(base_path + "pickles/task_1_data"):
    data = misc.load_pickle(base_path + "pickles/task_1_data")
    X, Y = data["X"], data["Y"]

    # DEBUG
    # X = X[:40000]
    # Y = Y[:40000]
else:
    X, Y = misc.import_csv_data(base_path + data_path, has_headers=True)
    misc.save_pickle(base_path + "pickles/task_1_data", {"X": X, "Y": Y})

# Print data shapes
print("Shapes")
print(X.shape)
print(Y.shape)

# Normalize (min values are always 0, so just divide by max per feature to get max == 1)
# Since the dataset is so big, I normalize before splitting because I use enough variety in the features
print("\nNormalization")
print(np.max(X, axis=0))
print(np.min(X, axis=0))
normalization_by_feature = np.max(X, axis=0)
X = X / normalization_by_feature
assert (len(np.unique(np.max(X, axis=0))) == 1)
misc.save_pickle(base_path + "pickles/task_1_max.py", normalization_by_feature)

# Split
print("\nSplit dataset")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Model selection
print("\nModel selection")

print("Random Forest")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
rf.fit(X_train, Y_train)
pred = rf.predict(X_test)
print(metrics.regression_error(rf.predict(X_train), Y_train))
print(metrics.regression_error(pred, Y_test))

print("\nMLP")
model = task_1.get_MLP()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=base_path + "checkpoints/task_1/model",
                                                               save_weights_only=True, save_best_only=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanAbsoluteError())
model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_test, Y_test), callbacks=[model_checkpoint_callback])

