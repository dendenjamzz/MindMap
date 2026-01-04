import sys
import json
from transformers import pipeline
import random

print("Loading NLP model...")  # ✅ Debug: See if model starts loading
sys.stdout.flush()

# Load the NLP model
nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("Model loaded successfully.")  # ✅ Debug: See if model loads
sys.stdout.flush()

def analyze_words(words):
    words_list = words.split(",")  # Split input words by comma
    words_list = [word.strip() for word in words_list]

    print("Processing words:", words_list)  # ✅ Debug
    sys.stdout.flush()

    # Example of randomly generated links for demo
    links = []
    nodes = [{"id": word} for word in words_list]

    for i in range(len(words_list)):
        for j in range(i + 1, len(words_list)):
            if random.random() > 0.5:  # Simulate a random connection
                links.append({"source": words_list[i], "target": words_list[j]})

    # Generate some suggested words for demo purposes
    suggested_words = [random.choice(words_list) + "ly", random.choice(words_list) + "ness"]

    result = {
        "nodes": nodes,
        "links": links,
        "suggested_words": suggested_words
    }
    
    print(json.dumps(result))
    sys.stdout.flush()

if __name__ == "__main__":
    print("Script started...")  # ✅ Debug
    sys.stdout.flush()

    words = sys.argv[1]
    analyze_words(words)

    print("Script finished.")  # ✅ Debug
    sys.stdout.flush()
