from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import pickle
import json
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Model, Sequential, save_model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


# Load GloVe embeddings
# Update with your GloVe file path
glove_file_path = 'embeddings\glove.6B.300d.txt'
glove_embeddings = load_glove_embeddings(glove_file_path)
# Load your JSON file with questions
with open('train.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Assuming your JSON is a list of dictionaries
questions = [item['question'] for item in data]
# Print out the lengths of your questions
# lengths = [len(question.split()) for question in questions]
# print(lengths)
max_length = max([len(word_tokenize(q)) for q in questions])
# print(max_length)


def encoding_data(questions, sequence_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions)

    sequences = tokenizer.texts_to_sequences(questions)
    sequences = pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='pre')

    vocab_size = len(tokenizer.word_counts) + 1  # Size of the vocabulary

    data_x = sequences[:, :-1]
    data_y = sequences[:, -1]
    data_y = to_categorical(data_y, num_classes=vocab_size)

    words_to_index = tokenizer.word_index

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_x, data_y, vocab_size, words_to_index


class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(
            input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name='att_bias', shape=(
            input_shape[-2], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


def build_model_with_glove(sequence_length, unit1, vocab_size, words_to_index, embedding_dim=300):
    # Create an embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in words_to_index.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Initialize the embedding layer with GloVe embeddings
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                input_length=sequence_length-1, trainable=False)

    # Build the model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(unit1, return_sequences=True)))
    model.add(Attention())
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        learning_rate=0.001), metrics=['accuracy'])
    return model

# model = build_model(sequence_length, unit1, vocab)
# model.summary()


def train_model_for_length(sequence_length, unit1, epochs):
    data_x, data_y, vocab_size, words_to_index = encoding_data(
        questions, sequence_length)

    if data_x.shape[1] != sequence_length - 1:
        print(
            f"Error: Sequence length of data ({data_x.shape[1]}) does not match expected sequence length ({sequence_length - 1}).")
        return

    print(data_x.shape)
    print(data_y.shape)

    model = build_model_with_glove(
        sequence_length, unit1, vocab_size, words_to_index)
    model.summary()

    filepath = f"lstmatt_len{sequence_length}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = model.fit(data_x, data_y, batch_size=128,
                        epochs=epochs, callbacks=callbacks_list)
    save_model(model, f'final_model_len{sequence_length}.h5')


# Define your sequence lengths, units, and epochs
sequence_lengths = [max_length]

# sequence_lengths = [10, 11, 12, 13, 14, 15]  # Adjust these values as needed
unit1_values = [32, 64, 64]   # Add more units as needed
epochs_values = [425, 450, 500]  # Add more epochs as needed

# Now you can use these lists in your loop
for seq_len, unit1, epochs in zip(sequence_lengths, unit1_values, epochs_values):
    train_model_for_length(seq_len, unit1, epochs)
