from textblob import TextBlob
import time

texts_to_analyze = [
    "I absolutely love the new Artificial Intelligence curriculum! It is so helpful.",
    "This bug is highly frustrating. The code keeps breaking every time I run it.",
    "The movie was okay, nothing special, but not terribly bad either.",
    "Wow! That was incredibly amazing! Best experience of my life!"
]

print("Initializing Conversational Sentiment Agent...\n")
time.sleep(1)

for i, text in enumerate(texts_to_analyze):
    blob = TextBlob(text)
    
    print(f"--- Analysis {i+1} ---")
    print(f"User Input: \"{text}\"")
    
    # Generate Tokens
    print(f"Tokenized: {blob.words}")
    
    # Calculate Sentiment
    # Polarity is [-1.0, 1.0] representing [Negative, Positive]
    polarity = blob.sentiment.polarity
    
    classification = "Neutral 😐"
    if polarity > 0.1:
        classification = "Positive 🙂"
    elif polarity < -0.1:
        classification = "Negative 😠"
        
    print(f"Calculated Polarity: {polarity:.2f} -> Result: {classification}")
    print("-" * 30 + "\n")
    time.sleep(1)

print("Agent finished analyzing corpus.")
