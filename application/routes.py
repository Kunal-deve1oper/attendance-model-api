from application import app,db
from retinaface import RetinaFace
from PIL import Image
from flask import request,jsonify
from application import model
from bson.objectid import ObjectId
from io import BytesIO
import json
import torch


# tested
# route to handle the student image upload
@app.route("/image/individual",methods=["POST"])
def insert_embeddings():
    collection = db["student_img"]
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image provided'}), 400
    image_stream = BytesIO(file.read())
    image_data = Image.open(image_stream).convert("RGB")
    json_data = request.form.get('jsonData')
    userData = {}
    if json_data:
        try:
            json_data_dict = json.loads(json_data)
            userData = json_data_dict
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON data provided'}), 400
    embeddings = model.getEmbeddings(image_data)
    if embeddings is None :
        return jsonify({'error': 'Upload a good image'}), 400
    embeddings_list = embeddings.squeeze().tolist()
    data = {
        "classroom_id": ObjectId(userData['classroomId']),
        "student_id": ObjectId(userData['studentId']),
        "embedding": embeddings_list
    }
    try:
        res = collection.insert_one(data)
    except Exception as e:
        return jsonify({'error': e})
    return jsonify({"msg": "Success"}), 202


# not tested
# route to handle the attendance
@app.route("/image/attendance", methods=["POST"])
def attendance():
    collection = db["student_img"]
    json_data = request.form.get('jsonData')
    userData = {}
    if json_data:
        try:
            json_data_dict = json.loads(json_data)
            userData = json_data_dict
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON data provided'}), 400
    response = requests.get(userData['url'])
    image = Image.open(BytesIO(response.content))
    resp = RetinaFace.detect_faces(url)
    faces = extract_faces(image, resp)
    res = collection.find({'classroom_id': ObjectId(userData['classroomId'])})
    cache_new = []
    for info in faces:
        embedding = model.getEmbeddings(info)
        if embedding is None:
            continue
        cache_new.append(embedding)
    verified = []
    for data in res:
        embedding = model.reconvert_embeddings(data.get('emmbedding'))
        items = {
            "studentId": data.get('student_id'),
            "embedding": embedding
        }
        verified.append(items)
    present = []
    for i in cache_new:
        temp = 5
        closest_face = None
        for j in verified:
            distance = (i - j['embedding']).norm().item()
            if distance < temp:
                temp = distance
                closest_face = j['studentId']
        if closest_face:
            present.append(closest_face)
    
    # will handle the csv
    
    return jsonify({"msg": "Success", "data": present})