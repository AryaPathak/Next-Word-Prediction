import numpy as np
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Input
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

import sys
import io

# Check if running in a Jupyter Notebook
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter Notebook.")
else:
    # Redirect stdout to handle Unicode characters properly (only needed in some environments)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Loading the data
path = '1661-0.txt'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

# Split dataset into each word
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# Unique words
unique_words = np.unique(words)
unique_word_index = {word: i for i, word in enumerate(unique_words)}
indices_word = {i: word for word, i in unique_word_index.items()}  # Reverse mapping


# Feature Engineering
WORD_LENGTH = 5  # No. of previous words to consider to predict next word
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
    

# Array X for storing features and Y for storing labels
X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_word in enumerate(prev_words):
    for j, word in enumerate(each_word):
        X[i, j, unique_word_index[word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1


 # Building RNN
model = Sequential()
model.add(Input(shape=(WORD_LENGTH, len(unique_words))))  # Add Input layer
model.add(LSTM(128))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))


 # Training the model
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

# Saving the Model
model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb")) 

# Testing
def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    words = text.split()  # Split input text into words
    
    for t, word in enumerate(words[:WORD_LENGTH]):
        if word in unique_word_index:  # Check if the word is in the vocabulary
            x[0, t, unique_word_index[word]] = 1
    return x


# Example input for testing
prepared_input = prepare_input("It is not a lack")
print(prepared_input)

# Function to return samples
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)  # Convert to log space
    exp_preds = np.exp(preds)  # Exponentiate
    preds = exp_preds / np.sum(exp_preds)  # Normalize
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

# Function for next word prediction
def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_word[idx] for idx in next_indices]  # Return the predicted words

quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]



for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print() 




