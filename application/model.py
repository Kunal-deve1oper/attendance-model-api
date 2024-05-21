from facenet_pytorch import MTCNN,InceptionResnetV1
import torch
from PIL import ImageDraw,ImageFont,Image
from io import BytesIO

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

def extract_faces(img, face_data):
    faces = []
    for face_info in face_data:
        if face_info['confidence'] < .96:
            continue
        x1, y1, w, h = map(int, face_info['box'])
        x2 = x1 + w
        y2 = y1 + h
        face_img = img.crop((x1, y1, x2, y2))
        byte_stream = BytesIO()
        face_img.save(byte_stream, format='JPEG')
        byte_stream.seek(0)
        loaded_img = Image.open(byte_stream)
        faces.append({"face_img": loaded_img, "box": face_info['box']})
        
    return faces

def draw_box(img,data):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=35)
    except IOError:
        font = ImageFont.load_default()

    for info in data:
        x, y, width, height = info['box']
        text = info['name']
        draw.rectangle([x, y, x + width, y + height], outline='green', width=2)
        text_bbox = draw.textbbox((x, y), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x, y - text_height - 2, x + text_width, y], fill='green')
        draw.text((x, y - text_height - 2), text, fill='white', font=font)