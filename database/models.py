from .db import db


class Question(db.Document):
    Question = db.StringField(required=True)
    Answers = db.StringField( required=True)
