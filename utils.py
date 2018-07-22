import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()


def clean_sentence(sentence):
    removed_markup = BeautifulSoup(sentence, 'html.parser').get_text()
    removed_punctuation = re.sub(r'[^a-zA-Z]', ' ', removed_markup)
    tokens = removed_punctuation.lower().split()
    removed_stopwords = [w for w in tokens if w not in stopwords.words('english')]
    lemmatized = [lemmatizer.lemmatize(w) for w in removed_stopwords]
    return ' '.join(lemmatized)


def extract_features(arr):
    vectorizer = CountVectorizer(analyzer='word', max_features=5000)
    features = vectorizer.fit_transform(arr)
    return features.toarray()
