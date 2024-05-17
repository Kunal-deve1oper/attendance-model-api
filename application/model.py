from facenet_pytorch import MTCNN,InceptionResnetV1
import torch

resnet = InceptionResnetV1(pretrained='casia-webface').eval()

mtcnn = MTCNN()

# function to generate embeddings
def getEmbeddings(image_data):
    w, h = image_data.size
    if w < 60 or h < 60:
        return None
    aligned = mtcnn(image_data)
    if aligned is None:
        return None
    aligned = aligned.unsqueeze(0)
    embeddings = resnet(aligned).detach()
    return embeddings

# function to convert the list to correct tensor format
def reconvert_embeddings(embeddings):
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
    return embeddings_tensor

# function to return extracted faces from group image
def extract_faces(img, face_data):
    faces = []
    for face_info in face_data:
        if face_info['confidence'] < .96:
            continue
        # Extracting face coordinates
        x1, y1, w, h = map(int, face_info['box'])
        x2 = x1 + w
        y2 = y1 + h
        # Cropping the face region
        face_img = img.crop((x1, y1, x2, y2))
        # Append cropped face image to list
        faces.append(face_img)
    return faces