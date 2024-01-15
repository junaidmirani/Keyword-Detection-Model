import re
import pickle
import numpy as np
from lstm import Attention
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
try:
    model = load_model('final_model_len20.h5', custom_objects={
                       'Attention': Attention})
    # model = load_model('lstmatt_len20.hdf5', custom_objects={'Attention': Attention})

except Exception as e:
    print(f"Error loading the model: {e}")
    exit()


def extra_space(text):
    new_text = re.sub("\s+", " ", text)
    return new_text


def sp_charac(text):
    new_text = re.sub("[^0-9A-Za-z ]", "", text)
    return new_text


def tokenize_text(text):
    new_text = word_tokenize(text)
    return new_text


def get_top_predictions(predictions, tokenizer, k=10):
    top_indices = predictions.argsort()[-k:][::-1]
    filtered_predictions = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, 'Unknown')
        prob = predictions[idx]
        if word != 'Unknown':
            filtered_predictions.append((word, prob))
    return filtered_predictions[:k]


def predict_next(tokenizer, model, input_text, k=10):
    cleaned_text = extra_space(input_text)
    cleaned_text = sp_charac(cleaned_text)
    tokenized = tokenize_text(cleaned_text)

    max_sequence_length = 20  # Replace with your actual max_length if different
    if len(tokenized) <= max_sequence_length:
        padded_sequence = pad_sequences([tokenizer.texts_to_sequences(
            [tokenized])[0]], maxlen=max_sequence_length - 1, truncating='pre')

        predictions = model.predict(padded_sequence)[0]

        print("Predictions for a known sequence:")
        top_predictions = get_top_predictions(predictions, tokenizer, k)
        for word, prob in top_predictions:
            print(f"{word}: {prob:.4f}")

    else:
        partial_input = ' '.join(tokenized[-max_sequence_length + 1:])
        print(
            f"Input exceeded max length. Predicting based on last {max_sequence_length - 1} words: {partial_input}")
        predict_next(tokenizer, model, partial_input, k)


# Example usage:
input_word = input("Enter a word or phrase: ")
predict_next(tokenizer, model, input_word, k=10)
