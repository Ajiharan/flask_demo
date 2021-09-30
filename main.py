from flask import Flask
from database.db import initialize_db
from flask_restful import Api
from resources.routes import initialize_routes
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY")
api = Api(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb://localhost/restdb'
}

initialize_db(app)
initialize_routes(api)

if __name__=='__main__':
    app.run(debug=True)