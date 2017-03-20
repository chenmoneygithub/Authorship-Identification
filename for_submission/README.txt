# Authorship-Identification
CS224N final project
team member: cqian23, th7, zhangrao

## Dataset
1. Put dataset (C50 or gutenberg) in dataset folder
2. Preprocess it using scripts in utils folder (e.g. create_sent_set_C50.py)
3. The result is a .pkl file and add its path to the model

## Running
Final models are:
1. RNN_average_model.py (sentence level model, gru)
2. RNN_sent_model_embed.py (article level model, with option gru or lstm cell in config)
3. siamese_network.py (article level siamese network model)

## Glove Word Vectors Directory
xxx/Authorship-Identification
xxx/data/glove/*.txt
