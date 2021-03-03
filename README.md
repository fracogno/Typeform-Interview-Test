# Typeform-Interview-Test
Solution of two problems for the Typeform Interview first stage.

## Docker
### Start docker compose for Flask to train / run prediction
sudo docker-compose up -d

### Enter docker container to train
> sudo docker exec -it ml /bin/bash

## Task 1

###
### Run prediction
> http://188.166.213.241:5000/typeform/task_1

SUBMISSIONS / VIEWS is regression problem, values are in the range [0,1].

Models cannot even overfit!! => HYPOTHESIS : Same rows with different regression labels ??
Double check with the next command:
> cat ml/dataset/completion_rate.csv | cut -d "," -f 4- | sort | uniq -d | wc

The above command: displays the file, selects only the features columns, sort them and see whether there are duplicates => There are!!

CONCLUSION : 

