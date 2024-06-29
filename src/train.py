import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize lemmatizer and label encoder
lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()

# Load the intents file
with open('data/intents.json') as file:
    data = json.load(file)

# Extract patterns and tags
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = sorted(set([lemmatizer.lemmatize(word.lower()) for pattern in patterns for word in nltk.word_tokenize(pattern)]))

# Encode tags
tags = sorted(set(tags))
labels = label_encoder.fit_transform(tags)

# Create training data
training_sentences = []
training_labels = []

for i, pattern in enumerate(patterns):
    word_list = nltk.word_tokenize(pattern)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    bag = [1 if word in word_list else 0 for word in words]
    training_sentences.append(bag)
    training_labels.append(labels[i])

training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(training_sentences[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(tags), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_sentences, training_labels, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Training complete.")
