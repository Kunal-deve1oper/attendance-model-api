from application import app,db,detector
from PIL import Image
from flask import request,jsonify
from application import model
from bson.objectid import ObjectId
from io import BytesIO
import json
import base64
import torch
import requests
import numpy as np
from datetime import datetime


# tested
# route to handle the student image upload
@app.route("/image/individual",methods=["POST"])
def insert_embeddings():
    collection = db["embeddings"]
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
    temp = collection.find_one({"student_id": userData['studentId']})
    if temp:
        if userData['classroomId'] in temp['code']:
            return jsonify({"msg": "Already enrolled"}), 409
        result = collection.update_one({"_id": temp['_id']},{'$push':{'code':userData['classroomId']}})
    else:
        data = {
            "code": [userData['classroomId']],
            "student_id": userData['studentId'],
            "name": userData['name'],
            "embedding": embeddings_list
        }
        try:
            res = collection.insert_one(data)
            student = db['students']
            student.update_one({"_id": ObjectId(userData['studentId'])},{'$set':{'embeddings_id': str(res.inserted_id)}})
        except Exception as e:
            return jsonify({'error': e}), 500
    return jsonify({"msg": "Success","embeddingId": str(res.inserted_id)}), 202


# route to handle enrolling of a student whose embeddings is already present
@app.route("/enroll",methods=["POST"])
def add_code():
    collection = db["embeddings"]
    data = request.json
    if data.get('classroomId') is None:
        return jsonify({"error": "classroom id not found"}), 400
    if data.get('studentId') is None:
        return jsonify({"error": "student id not found"}), 400
    temp = collection.find_one({"student_id": data.get('studentId')})
    if temp:
        if data.get('classroomId') in temp['code']:
            return jsonify({"msg": "Already enrolled"}), 409
        result = collection.update_one({"_id": temp['_id']},{'$push':{'code': data.get("classroomId")}})
    else:
        return jsonify({"msg": "Enter Correct student id"}), 400
    return jsonify({"msg": "Success"}), 202


# tested
# route to handle the attendance
@app.route("/image/attendance", methods=["POST"])
def attendance():
    collection = db["embeddings"]
    data = request.json
    if data.get('url') is None:
        return jsonify({"error": "url not found"}), 400
    if data.get('classroomId') is None:
        return jsonify({"error": "classroom id not found"}), 400
    url = data.get('url')
    classroomId = data.get('classroomId')
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    img = np.array(image)
    resp = detector.detect_faces(img)
    faces = model.extract_faces(image, resp)
    res = collection.find({'code': classroomId})
    if not res:
        return jsonify({"error": "Classroom not found"}), 404
    cache_new = []
    for info in faces:
        embedding = model.getEmbeddings(info['face_img'])
        if embedding is None:
            continue
        cache_new.append({"embedding":embedding,"box":info['box']})
    verified = []
    for data in res:
        embedding = model.reconvert_embeddings(data.get('embedding'))
        items = {
            "studentId": data.get('student_id'),
            "name": data.get('name'),
            "embedding": embedding
        }
        verified.append(items)
    present = []
    boxing = []
    for i in cache_new:
        temp = 5
        closest_face = None
        for j in verified:
            distance = (i['embedding'] - j['embedding']).norm().item()
            if distance < temp:
                temp = distance
                closest_face = j['studentId']
                name = j['name']
        if closest_face:
            present.append(str(closest_face))
            boxing.append({"name":name,"box":i['box']})
    
    ans = []
    for data in verified:
        if str(data['studentId']) in present:
            temp = {
                "studentId": str(data['studentId']),
                "name": data['name'],
                "present": 1
            }
            ans.append(temp)
        else:
            temp = {
                "studentId": str(data['studentId']),
                "name": data['name'],
                "present": 0
            }
            ans.append(temp)
    
    cumulative = db['cumulatives']
    cumulative_attendance = {
        "classroomId": classroomId,
        "date": datetime.now(),
        "attendance": ans
    }
    try:
        res = cumulative.insert_one(cumulative_attendance)
    except Exception as e:
        return jsonify({"msg": e}), 500

    model.draw_box(image,boxing)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Return the JSON response with the image included
    return jsonify({
        "msg": "Success",
        "date": datetime.now().date(),
        "data": ans,
        "image": img_str
    }), 200