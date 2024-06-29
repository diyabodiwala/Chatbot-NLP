import json
import random
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Load the intents file
with open('data/intents.json') as file:
    data = json.load(file)

# Initialize lemmatizer and label encoder
lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()

# Load the model
model = tf.keras.models.load_model('chatbot_model.h5')

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

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': tags[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

print("Chatbot is running!")

while True:
    message = input("You: ")
    intents = predict_class(message)
    response = get_response(intents, data)
    print(f"Bot: {response}")
