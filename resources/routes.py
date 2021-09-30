from .questions import QuestionApi, QuestionsApi
from .userAuth import LoginUser,SignupUser

def initialize_routes(api):
    api.add_resource(QuestionsApi, '/api/questions')
    api.add_resource(QuestionApi, '/api/questions/<id>')
    api.add_resource(SignupUser, '/api/user/signup')
    api.add_resource(LoginUser, '/api/user/login')
