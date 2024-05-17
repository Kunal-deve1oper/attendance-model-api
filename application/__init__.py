from flask import Flask
from pymongo import MongoClient
from flask_cors import CORS
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb+srv://twitterabhinil:botkunal@attendancecluster.yctmyxz.mongodb.net/")
db = client["attendanceSystem"]
detector = MTCNN()

from application import routes