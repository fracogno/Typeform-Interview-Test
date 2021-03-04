# Typeform-Interview-Test
Solution of two problems for the Typeform Interview first stage.

## Docker
### Start docker compose for Flask to train / run prediction (ONLY DONE IN MY CLOUD, client doesn't need this)
sudo docker-compose up -d

### Enter docker container to train
> sudo docker exec -it ml /bin/bash

## Task 1
### Goal : Predict SUBMISSIONS / VIEWS (regression problem). Values are in the range [0,1].

### Modus operandi
Train Random forest / MLP => ERROR is ~ 0.17 on test set. It means by average the prediction is 17% of the actual ground truth value (% because values between 0-1).

### Run prediction in the cloud (hosted with Docker)
> python3 ml/src/predict.py

### N.B.
Models cannot even overfit 100%!! => HYPOTHESIS : Same rows with different regression labels
Double check with the next command:
> head -n 50000 ml/dataset/completion_rate.csv | cut -d "," -f 4- | sort | uniq -d | wc
The above command: displays the file, selects only the features columns, sort them and see whether there are duplicates => There are!!

CONCLUSION :
The accuracy isn't too bad! However, having same rows with different labels it doesn't help....

## Task 2
### Goal : Cluster same type of questions (Unsupervised learning)

### Modus operandi
Train a Encoder-Decoder model with attention as an autoencoder (reconstruct the input sequence). Moreover, I use pretrained embeddings from GloVe.
Once the model has been trained, I throw the decoder away and just use the encoder. I do this to decoder my sentences into a fixed amount of numbers (in my case 16 numbers). After that I encoded my sentence into 16 numbers, I apply PCA to get 3 numbers out of this.
Finally, I plot in 3D using Tensorboard the sentences. It can be noticed that similar sentences cluster together in 3D space. This is how the unsupervised clustering is done!

### Run training
> python3 ml/src/task_2.py

### Run predictions and PCA
> python3 ml/src/predict_task_2.py

### TO VISUALIZE sentences in 3D space
> tensorboard --logdir=ml/checkpoints/task_2/
> http://localhost:6006/#projector

### N.B.
I have put up a quick model. Many improvements could be made! Can be discussed orally.
