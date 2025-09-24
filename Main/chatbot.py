import random
import json
import pickle
import numpy as np

import nltk
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "intents.json"))

with open(INTENTS_PATH, "r", encoding="utf-8") as file:
    intents = json.load(file)


lemmatizer = WordNetLemmatizer()


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_webseedermodel.keras')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    return [{'intent': classes[max_index], 'probability': str(res[max_index])}]

def get_response(ints, intents_json):
    if len(ints) == 0:  
        return random.choice([
            "🤔 Sorry, I didn’t understand that. Could you rephrase?",
            "⚡ I’m still learning! Can you try asking differently?",
            "🙋 You can ask me about our services like web development, app development, AI solutions, and more!"
        ])
    
    tag = ints[0]['intent']
    prob = ints[0]['probability']

    if float(prob) < 0.4:  
        return random.choice([
            "🤔 Hmm, I’m not sure what you mean.",
            "⚡ Can you please clarify?",
            "🙋 Try asking me about our services like web dev, app dev, or AI solutions!"
        ])

    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

print("chatBot is running!")

while True:
    message = input("Enter any message")
    ints = predict_class (message)
    print("DEBUG Predictions:", ints)
    res = get_response (ints, intents)
    print ("bot:",res)



