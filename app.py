from flask import Flask, request, Response,jsonify
from database.db import initialize_db
from train_model.train import *
from database.models import Question
import json
import pandas as pd

import csv

app = Flask(__name__)
app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb://localhost/restdb'
}

initialize_db(app)

@app.route("/")
def import_csv():
    result = Question.objects().only('Question')
    lst = list(map(lambda res: {'Question': res['Question']}, result))
    df = pd.read_json(json.dumps(lst))
    df.to_csv('test1.csv', header=['questions'], index=False, sep="\\", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    # questions = result.to_json();
    train_question_model()
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