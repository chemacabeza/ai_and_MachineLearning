<div align="center">
  <img src="cover.png" alt="Fake News Detection Cover" width="800"/>
</div>

# Chapter 16: Fake News Detection

**🎯 The Big Goal:** Learn how NLP and machine learning can automatically distinguish real news from fake news by analyzing linguistic patterns, writing style, and source credibility signals.

## Core Concepts

Misinformation spreads 6x faster than real news on social media. **Fake News Detection** uses AI to automatically flag unreliable content before it goes viral.

### How AI Detects Fake News

1. **Linguistic Features:** Fake news uses more sensational language ("SHOCKING!", "You won't BELIEVE"), more capital letters, more exclamation marks, and more emotional words.
2. **Source Analysis:** Real news comes from established outlets with editorial standards. AI can learn source reputation patterns.
3. **Propagation Patterns:** Fake news spreads differently — it often originates from bot networks and spreads through echo chambers.
4. **Fact Verification:** Advanced systems cross-reference claims against knowledge bases.

### TF-IDF: The Workhorse of Text Classification

**Term Frequency-Inverse Document Frequency** converts text into numbers:
- **TF:** How often a word appears in one document.
- **IDF:** How rare a word is across all documents.
- Words that are frequent in one document but rare overall get high TF-IDF scores — they're distinctive.

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: Can AI completely solve the fake news problem?</summary>

No. AI can flag likely fake content, but the problem is fundamentally social, not technical. Sophisticated disinformation blends true and false information, uses real sources selectively, and exploits cognitive biases. Satire and opinion pieces add ambiguity. AI is a tool in the fight against misinformation, but media literacy, editorial standards, and platform policies are equally important.
</details>

<details>
<summary>💡 View Answer: What is the danger of false positives in fake news detection?</summary>

Incorrectly flagging real news as fake is dangerous — it can silence legitimate journalism, reinforce censorship, and erode trust in media. This is why real-world systems use confidence thresholds and human review rather than fully automated removal.
</details>

---

## 🐳 Hands-On Exercise: TF-IDF Fake News Classifier

### Step 1: Build
```bash
cd exercise
docker build -t ch16-fake-news .
```

### Step 2: Run
```bash
docker run --rm ch16-fake-news
```

### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install numpy scikit-learn
COPY fake_news.py /app/
CMD ["python", "fake_news.py"]
```
