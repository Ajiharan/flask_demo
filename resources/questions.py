from flask import Response, request
from database.models import Question
from flask_restful import Resource
from flask import Flask, request, Response,jsonify
from database.db import initialize_db
from train_model.train import *
from database.models import Question
import json
import pandas as pd
import csv

class QuestionsApi(Resource):
    def get(self):
        result = Question.objects().only('Question')
        lst = list(map(lambda res: {'Question': res['Question']}, result))
        df = pd.read_json(json.dumps(lst))
        df.to_csv('test1.csv', header=['questions'], index=False, sep="\\", encoding='utf-8',
                  quoting=csv.QUOTE_NONNUMERIC)
        # questions = result.to_json();
        train_question_model()
        return Response("model trained sucessfully", mimetype="application/json", status=200)

    def post(self):
        body = request.get_json()
        question, similarity, accuracy = input_question(body['question'])
        return {'my_question': question, 'similarity_question': similarity, 'accuracy': accuracy * 100}, 200


class QuestionApi(Resource):
    def put(self, id):
        body = request.get_json()
        Question.objects.get(id=id).update(**body)
        return '', 200

    def delete(self, id):
        movie = Question.objects.get(id=id).delete()
        return '', 200

    def get(self, id):
        movies = Question.objects.get(id=id).to_json()
        return Response(movies, mimetype="application/json", status=200)