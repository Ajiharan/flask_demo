from flask import Flask, request, Response,jsonify
from database.db import initialize_db
from model.QuestionModel import *
from database.models import Question
import json
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
import ast
import csv
app = Flask(__name__)
app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb://localhost/restdb'
}

initialize_db(app)



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
    # print('question',QuestionModel.fetch_data)

    query_vect = QuestionModel.tfidf_vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, QuestionModel.tfidf_matrix)

    max_similarity = np.argmax(similarity, axis=None)

    return question,QuestionModel.fetch_data.iloc[max_similarity]['questions'],similarity[0, max_similarity]


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


def train_model():
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
    QuestionModel.fetch_data=fetch_data
    QuestionModel.tfidf_matrix=tfidf_matrix
    QuestionModel.tfidf_vectorizer=tfidf_vectorizer


@app.route("/")
def import_csv():
    result = Question.objects().only('Question')
    lst = list(map(lambda res: {'Question': res['Question']}, result))
    df = pd.read_json(json.dumps(lst))
    df.to_csv('test1.csv', header=['questions'], index=False, sep="\\", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    questions = result.to_json();
    train_model()
    return Response("model trained sucessfully", mimetype="application/json", status=200)

@app.route('/questions')
def get_questions():
    result = Question.objects().only('Question')
    questions=result.to_json()
    return Response(questions, mimetype="application/json", status=200)

@app.route('/ask',methods=['POST'])
def askQuestion():
    body=request.get_json()
    question,similarity,accuracy=input_question(body['question'])
    print('My question:', question)
    print('Similarity question found:', similarity)
    print('Accuracy: {:.2%}'.format(accuracy))
    return {'my_question':question,'similarity_question':similarity,'accuracy':accuracy*100}, 200

@app.route('/questions', methods=['POST'])
def add_question():
    body = request.get_json(force=True)
    movie =  Question(**body).save()
    id = movie.id
    if(id):
        return {'id': str(id)}, 200
    else:
        return {'error':''},400

@app.route('/questions/<id>', methods=['PUT'])
def update_question(id):
    body = request.get_json()
    Question.objects.get(id=id).update(**body)
    return 'updated', 200

@app.route('/questions/<id>', methods=['DELETE'])
def delete_question(id):
    question = Question.objects.get(id=id).delete()
    return 'deleted', 200

@app.route('/questions/<id>')
def get_question(id):
    questions = Question.objects.get(id=id).to_json()
    return Response(questions, mimetype="application/json", status=200)

if __name__ == '__main__':
    app.run(debug=True)