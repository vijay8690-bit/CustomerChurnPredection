import numpy as np
from keras_preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token='vijay')

docs = ['go india',
        'india india',
        'hip hip hurray',
        'jeetega bhai jeetega india jeetega',
        'baharat mata ki jai',
        'kolhi kolhi',
        'doni doni',
        'sachin sachin',
        'modi ji ki jai',
        'inquilab Zindabad'
        ]

tokenizer.fit_on_texts(docs)

# print(tokenizer.word_index)
# print(tokenizer.word_counts)

sequences = tokenizer.texts_to_sequences(docs)
print(sequences)