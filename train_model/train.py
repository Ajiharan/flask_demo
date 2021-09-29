import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib


def formatString(sen):
    formatQues=[]
    for word in sen.split():
        if len(doit(word))==0:
            formatQues.append(word)
        else:
             formatQues.append(doit(word))
    return ' '.join(formatQues);


def doit(text):
    matches=re.findall(r'\"(.+?)\"',text)
    if(len(matches)==0):
        return []

    return ",".join(matches)


def input_question(question):
    file = open('models/tfidf_vectorizer_model.pkl', "rb")
    file_matrix = open('models/tfidf_matrix_model.pkl', "rb")
    fetchDatas = open('models/fetch_data.pkl', "rb")

    # load the trained model
    trained_model = joblib.load(file)
    trained_model_matrix = joblib.load(file_matrix)
    fetch_data = joblib.load(fetchDatas)

    query_vect = trained_model.transform([question])
    similarity = cosine_similarity(query_vect, trained_model_matrix)

    max_similarity = np.argmax(similarity, axis=None)

    return question,fetch_data.iloc[max_similarity]['questions'],similarity[0, max_similarity]


def my_tokenizer(doc):
        stopwords_list = stopwords.words('english')

        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(doc)

        pos_tags = pos_tag(words)

        non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]

        non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

        lemmas = []
        for w in non_punctuation:
            if w[1].startswith('J'):
                pos = wordnet.ADJ
            elif w[1].startswith('V'):
                pos = wordnet.VERB
            elif w[1].startswith('N'):
                pos = wordnet.NOUN
            elif w[1].startswith('R'):
                pos = wordnet.ADV
            else:
                pos = wordnet.NOUN
            lemmas.append(lemmatizer.lemmatize(w[0], pos))

        return lemmas


def train_question_model():
    fetch_data = pd.read_csv('test1.csv')
    print(fetch_data.info())

    fetch_data = fetch_data[['questions']]
    fetch_data.head(10)

    fetch_data = fetch_data.drop_duplicates(subset='questions')
    fetch_data.head(10)

    fetch_data = fetch_data.dropna()
    fetch_data.shape

    tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(fetch_data['questions']))
    print(tfidf_matrix.shape)
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer_model.pkl')
    joblib.dump(tfidf_matrix, 'models/tfidf_matrix_model.pkl')
    joblib.dump(fetch_data, 'models/fetch_data.pkl')

# question,similarity,accuracy=input_question(formatString('What is RAM?'))
# print('My question:', question)
# print('Similarity question found:', similarity)
# print('Accuracy: {:.2%}'.format(accuracy))