import random
import json
import pickle
import numpy as np
import nltk
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "intents.json"))

with open(INTENTS_PATH, "r", encoding="utf-8") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_webseedermodel.keras")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence, threshold=0.2):
    # # Directly check for 'webseeder' keyword and override intent
    # if "webseeder" in sentence.lower():
    #     return [{"intent": "about_webseeder", "probability": "1.0"}]

    if "about webseeder" in sentence.lower():
        return [{"intent": "about", "probability": "1.0"}]
    if sentence.lower().strip() in ["service", "services"]:
        return [{"intent": "services", "probability": "1.0"}]

    bow = bag_of_words(sentence)
    if np.sum(bow) == 0:
        return []

    res = model.predict(np.array([bow]), verbose=0)[0]

    results = [
        {"intent": classes[i], "probability": str(prob)}
        for i, prob in enumerate(res)
        if prob > threshold
    ]

    results.sort(key=lambda x: float(x["probability"]), reverse=True)
    return results

def get_response(intents_list, intents_json):
    if not intents_list:
        return random.choice([
            "🤔 Sorry, I didn’t understand that. Could you rephrase?",
            "⚡ I’m still learning! Can you try asking differently?",
            "🙋 You can ask me about our services like web development, app development, AI solutions, and more!"
        ])

    tag = intents_list[0]["intent"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, something went wrong finding a response."

print("🤖 ChatBot is running! Type something...")

while True:
    message = input("You: ")
    if message.lower() in ["quit", "exit", "bye"]:
        print("🤖 Bot: Goodbye! Have a great day! 👋")
        break

    intents_predicted = predict_class(message)
    response = get_response(intents_predicted, intents)
    print("🤖 Bot:", response)
