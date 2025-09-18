# 🤖 Webseeder Chatbot

This is a custom **AI-powered chatbot** built using **Python, TensorFlow, and NLP techniques**.  
It is trained on a set of intents (`intents.json`) to answer queries about Webseeder’s services like **Web Development, App Development, Blockchain, and AI/ML**.

---

## 🚀 Features
- Intent-based responses (web, app, blockchain, AI/ML, contact info, etc.)
- Handles greetings, goodbyes, and contact email queries
- Default fallback responses for unknown queries
- Uses **relative paths** (works across different systems)
- Includes **NLTK downloads** for tokenization and lemmatization

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Webseeder-chatbot.git
   cd Webseeder-chatbot

2. **Install dependencies**
     pip install -r requirements.txt

3. **Download NLTK resources (run once)**
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')

4.**Train the model (only if not already trained)**
   python new.py

   This will generate:
    chatbot_webseedermodel.keras
    words.pkl
    classes.pkl
5.**Run the chatbot**
    python chatbot.py
