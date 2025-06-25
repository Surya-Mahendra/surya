import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

data = """Artificial intelligence is changing the world. It has transformed education, healthcare, and industry. The future will see even greater integration of AI in daily life."""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
vocab_size = len(tokenizer.word_index) + 1

sequences = []
tokens = tokenizer.texts_to_sequences([data])[0]
for i in range(1, len(tokens)):
    seq = tokens[:i+1]
    sequences.append(seq)

max_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
sequences = np.array(sequences)

X, y = sequences[:, :-1], to_categorical(sequences[:, -1], num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)

def generate_lstm_text(seed_text, n_words):
    result = []
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length-1, padding='pre')
        y_pred = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        word = ''
        for w, i in tokenizer.word_index.items():
            if i == y_pred:
                word = w
                break
        in_text += ' ' + word
        result.append(word)
    return seed_text + ' ' + ' '.join(result)

print(generate_lstm_text("Artificial intelligence", 10))
