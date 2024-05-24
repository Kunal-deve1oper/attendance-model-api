from flask import Flask
from pymongo import MongoClient
from flask_cors import CORS
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb+srv://twitterabhinil:botkunal@attendancecluster.yctmyxz.mongodb.net/")

# client = MongoClient("mongodb+srv://ghoshkunal106:?????@cluster0.6qg4grh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["attendanceSystem"]
# db = client["final"]
detector = MTCNN()

from application import routes