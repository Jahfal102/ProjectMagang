import cv2
import numpy as np
import base64
import os
import io
# from fer import FER
from typing import Annotated
from typing import Union
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, File, UploadFile, Request
from insightface.app import FaceAnalysis
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.logger import logger
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

model = FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))
app = FastAPI()

@app.get("/health")
def healthcheck():
    return {
        "success": True,
        "message": "OK"     
    }


@app.post("/api/face-detection")
async def face_detection(
    # image: Annotated[UploadFile, File()],
    request: Request, images: List[UploadFile] = File(...)
    ):
    # print("Images received:", images)
    # if not images:
    #     return {"error": "No images received"}
    # for img in images:
    #     print(f"Received file: {img.filename}")
    # return {"message": "Files received", "filenames": [file.filename for file in images]}
    # logger.info(f"Received files: {images}")

    # if not isinstance(images, list):
    #     images = [images]
    output_dir = "output"

    # ini pas di postman
    results = []
    for image in images:
        # Parse setiap image ke dalam OpenCV Mat
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        imageMat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Inference Face Detection dengan Insightface
        bbox_result = []
        faces = model.get(imageMat)
        for face in faces:
            # Gambar bounding box pada gambar
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(imageMat, (x1, y1), (x2, y2), (0, 255, 0), 2)
            bbox_result.append([x1, y1, x2, y2])

        # # Simpan gambar sementara dengan bounding box
        # output_filename = os.path.join(output_dir, f"output_{image.filename}")
        # cv2.imwrite(output_filename, imageMat)

        # # Tambahkan ke hasil
        # results.append({
        #     "filename": image.filename,
        #     "output_file": output_filename,
        #     "faces": bbox_result
        # })
        
        # Jika hanya ada satu gambar, kembalikan langsung sebagai stream
        _, img_encoded = cv2.imencode('.jpg', imageMat)
        img_bytes = io.BytesIO(img_encoded.tobytes())
        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=result.jpg"}
        )
    return {
        "success": True,
        "message": "Processed multiple images successfully",
        "results": results
    }

@app.post("/api/face-profile")
async def face_profile(
    image_profile: UploadFile = File(...)
    ):

    contents = await image_profile.read()
    nparr = np.frombuffer(contents, np.uint8)
    imageMat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    muka = model.get(imageMat)
    profiles = []

    for face in muka:
        profile = {
            "gender": "male" if face.gender > 0.5 else "female",  # Gender
            "age": int(face.age)  # Age
            # "emotion": face.kps.tolist() if hasattr(face, "kps") else "N/A"
            }
        profiles.append(profile)

    return {
        "success": True,
        "message": "Success profile face in image",
        "result": {
            "profiles": profiles
        }
    }

@app.post("/api/face-similarity")
async def face_similarity(
    image1: UploadFile = File(...), image2: UploadFile = File(...)
    ):

    contents1 = await image1.read()
    contents2 = await image2.read()
    nparr1 = np.frombuffer(contents1, np.uint8)
    nparr2 = np.frombuffer(contents2, np.uint8)

    imageMat1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    imageMat2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    faces1 = model.get(imageMat1)
    faces2 = model.get(imageMat2)

    if len(faces1) == 0 or len(faces2) == 0:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "No face detected in one or both images."}
        )
    
    # Ambil face embeddings dari wajah pertama
    embedding1 = faces1[0].embedding
    embedding2 = faces2[0].embedding

    # Hitung cosine similarity
    similarity_score = float(cosine_similarity([embedding1], [embedding2])[0][0] * 100)

    return {
        "success": True,
        "message": "Success calculate face similarity",
        "result": {
            "similarity_score": round(similarity_score, 2)
        }
    }