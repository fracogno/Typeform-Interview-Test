# Typeform-Interview-Test
Solution of two problems for the Typeform Interview first stage.

## Docker
sudo docker-compose up -d

## Task 1
	
SUBMISSIONS / VIEWS is regression problem, values are in the range [0,1].


Models cannot even overfit!! => HYPOTHESIS : Same rows with different regression labels ??
Double check with the next command:
> cat ml/dataset/completion_rate.csv | cut -d "," -f 4- | sort | uniq -d | wc

The above command: displays the file, selects only the features columns, sort them and see whether there are duplicates => There are!!

CONCLUSION : 

