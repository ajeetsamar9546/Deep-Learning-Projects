import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

reviews = [
    "I love this product! It's amazing.",
    "The product was okay, nothing special.",
    "I did not like this product at all.",
    "The quality of this product is great.",
    "Terrible product, I will never buy it again."
]

# Corresponding sentiment labels: 1 for positive, 0 for negative
labels = np.array([1, 0, 0, 1, 0])


# Tokenizing the text
tokenizer = Tokenizer(oov_token='<nothing>')
tokenizer.fit_on_texts(reviews)
# print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(reviews)

# Padding the sequences to ensure uniform input size
max_length = max([len(x) for x in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')


# Building the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32, input_length=max_length))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10)

# Test the model with a new review
new_review = ["This is the best purchase I have made."]
new_sequence = tokenizer.texts_to_sequences(new_review)
padded_new_sequence = pad_sequences(new_sequence, maxlen=max_length, padding='post')

# Predict sentiment
prediction = model.predict(padded_new_sequence)
print("Sentiment score:", prediction[0][0])




















