import pandas as pd
import re, nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("Loading data...")
fake_df = pd.read_csv('Fake.csv', usecols=['title','text'])
true_df = pd.read_csv('True.csv', usecols=['title','text'])

fake_df['label'] = 1
true_df['label'] = 0

df = pd.concat([fake_df, true_df], ignore_index=True)
df['content'] = df['title'] + ' ' + df['text']

print("Cleaning text...")
df['clean_text'] = df['content'].apply(clean_text)

X = df['clean_text']
y = df['label']

print("Training model...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# SAVE
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully ✅")