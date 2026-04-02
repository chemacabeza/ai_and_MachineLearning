<div align="center">
  <img src="cover.png" alt="Conversational AI & NLP Cover" width="800"/>
</div>

# Chapter 4: Conversational AI & Natural Language Processing

**🎯 The Big Goal:** Understand how machines process, tokenize, and extract sentiment from human text to behave like conversational agents.

## Core Concepts

Natural Language Processing (NLP) is the intersection of computer science and linguistics. It governs how computers read, decipher, understand, and make sense of human languages in a valuable way.

### Text Tokenization & Sentiment Analysis

Language is incredibly unstructured. We use sarcasm, slang, and double negatives. For a computer to process a sentence, it first must break it down:
1. **Tokenization**: Splitting a large text into smaller individual words or "tokens" so the machine can process them as discrete data points.
2. **Stopwords Removal**: Removing extremely common words (like 'the', 'is', 'a') that do not offer analytical value.
3. **Sentiment Analysis**: Using pre-trained lexicons or machine learning models to classify the emotional tone of text as Positive, Negative, or Neutral. 

Modern Conversational AI (like ChatGPT or Claude) uses highly advanced Deep Neural Networks (Transformers) to not only tokenize and derive sentiment but to predict the most contextually relevant next sequence of words!

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: What is a "Corpus" in NLP?</summary>

A **Corpus** is a large and structured set of texts (nowadays, usually electronically stored and processed). They are used to do statistical analysis and hypothesis testing, checking occurrences or validating linguistic rules. The internet is considered a massive corpus!
</details>

<details>
<summary>💡 View Answer: How does context affect sentiment analysis algorithms?</summary>

Highly negatively! If someone says "The movie was bad," that's negative. But what if they say "The movie was so bad it was good"? A rudimentary algorithm might flag the word "bad" and misclassify the sentiment. This is why advanced attention mechanisms in modern LLMs are necessary to grasp *contextual* meaning across multiple tokens.
</details>

---

## Hands-On Exercise: The Sentiment & Tokenizer Bot

In this exercise, you will run a simple script that acts like an early conversational agent by utilizing Python's `TextBlob` library to instantly analyze text input!

### Step 1: Build the Docker Environment
Navigate to the `exercise` folder and run:
```bash
cd exercise
docker build -t ch4-conversational-ai .
```

### Step 2: Run the NLP Script
```bash
docker run --rm ch4-conversational-ai
```

The script will automatically tokenize sample texts and display the calculated "Polarity" (Sentiment) of each text block!


### Source Code

```python
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
```
