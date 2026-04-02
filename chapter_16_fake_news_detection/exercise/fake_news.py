import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("=== Fake News Detection: TF-IDF + Classifier ===\n")

# Simulated news dataset
real_news = [
    "Scientists discover new treatment for heart disease in clinical trial",
    "Stock market rises 2% following positive economic data report",
    "New renewable energy plant opens providing power to 50000 homes",
    "Research shows regular exercise reduces risk of chronic disease",
    "International summit addresses climate change with new agreements",
    "University study finds link between sleep quality and productivity",
    "City council approves budget for new public transportation system",
    "Medical journal publishes peer reviewed study on vaccine efficacy",
    "Technology company reports quarterly earnings beating expectations",
    "Archaeological team discovers ancient artifacts dating back 3000 years",
]

fake_news = [
    "SHOCKING aliens secretly control world governments revealed insider",
    "This ONE weird trick will make you a millionaire overnight guaranteed",
    "BREAKING celebrity secretly a lizard person sources confirm",
    "Doctors HATE this miracle cure big pharma doesnt want you to know",
    "URGENT share before deleted government hiding truth from citizens",
    "You wont BELIEVE what scientists found will blow your mind",
    "SECRET footage proves moon landing was filmed in Hollywood studio",
    "AMAZING new pill lets you eat anything and lose 50 pounds instantly",
    "EXPOSED the real reason they dont want you to know the truth",
    "TERRIFYING discovery proves everything you know is a lie wake up",
]

texts = real_news + fake_news
labels = [0]*len(real_news) + [1]*len(fake_news)  # 0=real, 1=fake

vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=200)
model.fit(X, labels)

print("📊 Training Results:")
predictions = model.predict(X)
print(f"   Accuracy: {accuracy_score(labels, predictions)*100:.0f}%\n")

# Show most indicative words
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
fake_words = sorted(zip(coef, feature_names), reverse=True)[:5]
real_words = sorted(zip(coef, feature_names))[:5]

print("🔴 Top words indicating FAKE news:")
for score, word in fake_words:
    print(f"   {word:20s} (score: {score:+.3f})")

print("\n🟢 Top words indicating REAL news:")
for score, word in real_words:
    print(f"   {word:20s} (score: {score:+.3f})")

# Test on new headlines
test_articles = [
    "New study published in Nature reveals promising cancer treatment results",
    "SHOCKING you wont believe this secret cure doctors are hiding from you",
]
print("\n📰 Testing on new articles:")
for article in test_articles:
    X_new = vectorizer.transform([article])
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0]
    label = "🔴 FAKE" if pred == 1 else "🟢 REAL"
    print(f"   {label} ({max(prob)*100:.0f}% confidence): {article[:60]}...")

print("\n✅ The classifier learns linguistic patterns that distinguish fake from real news!")
