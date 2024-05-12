from flask import Flask
from pymongo import MongoClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb+srv://twitterabhinil:botkunal@attendancecluster.yctmyxz.mongodb.net/")
db = client["attendanceSystem"]

from application import routes