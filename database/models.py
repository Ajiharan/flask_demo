from .db import db
from flask_bcrypt import generate_password_hash, check_password_hash

class Question(db.Document):
    Question = db.StringField(required=True)
    Answers = db.StringField( required=True)

class User(db.Document):
    userRole=db.IntField(default=0)
    userName=db.StringField(required=True)
    email=db.StringField(required=True)
    password=db.StringField(required=True)

    def hash_user_password(self):
        self.password = generate_password_hash(self.password).decode('utf8')

    def check_user_password(self, oldPassword):
        return check_password_hash(self.password, oldPassword)

