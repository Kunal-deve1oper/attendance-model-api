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
def extract_faces(img, face_data, expansion_factor=1.07):
    faces = []
    for face_id, face_info in face_data.items():
        # Extracting face coordinates
        x1, y1, x2, y2 = map(int, face_info["facial_area"])
        # Calculating expanded bounding box
        width = x2 - x1
        height = y2 - y1
        expand_width = int(width * expansion_factor)
        expand_height = int(height * expansion_factor)
        expanded_x1 = max(0, x1 - (expand_width - width) // 2)
        expanded_y1 = max(0, y1 - (expand_height - height) // 2)
        expanded_x2 = min(img.width, x2 + (expand_width - width) // 2)
        expanded_y2 = min(img.height, y2 + (expand_height - height) // 2)
        # Cropping the expanded face region
        face_img = img.crop((expanded_x1, expanded_y1, expanded_x2, expanded_y2))
        # Append cropped face image to list
        faces.append(face_img)
    return faces