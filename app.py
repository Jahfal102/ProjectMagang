import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os

# Specify the path where you saved the models
# This path should be relative to your project or absolute
model_dir = "ProjectMagang/main.py"  # Make sure this path points to the directory with the models

# Ensure that the model directory exists (it should already contain the models after downloading)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize the InsightFace model with the specified root for the models
model = FaceAnalysis(providers=['CPUExecutionProvider'], root=model_dir)
model.prepare(ctx_id=0, det_size=(640, 640))

# Streamlit App
st.title("Face Analysis Application")

# Face Detection
st.header("Face Detection")
uploaded_file = st.file_uploader("Upload an image for face detection", type=["jpg", "jpeg", "png"], key="face-detection")
if uploaded_file is not None:
    # Read and process the image
    contents = uploaded_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imageMat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform face detection
    faces = model.get(imageMat)
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(imageMat, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with bounding boxes
    st.image(cv2.cvtColor(imageMat, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_column_width=True)

# Face Profile
st.header("Face Profile")
uploaded_profile_file = st.file_uploader("Upload an image for face profiling", type=["jpg", "jpeg", "png"], key="face-profile")
if uploaded_profile_file is not None:
    # Read and process the image
    contents = uploaded_profile_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imageMat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(imageMat, cv2.COLOR_BGR2RGB)

    # Perform face profiling
    faces = model.get(imageMat)
    st.image(image_rgb, caption="Uploaded Image", width=300)  # Display the image
    if faces:  # Check if at least one face is detected
        face = faces[0]  # Take the first detected face
        gender = "Male" if face.gender > 0.5 else "Female"
        age = int(face.age)
        st.write(f"Gender: {gender}  Age: {age}")
    else:
        st.error("No face detected in the uploaded image.")

# Face Similarity
st.header("Face Similarity")
uploaded_file1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"], key="similarity-image1")
uploaded_file2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"], key="similarity-image2")

if uploaded_file1 and uploaded_file2:
    # Read and process the first image
    contents1 = uploaded_file1.read()
    nparr1 = np.frombuffer(contents1, np.uint8)
    imageMat1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for display
    image_rgb1 = cv2.cvtColor(imageMat1, cv2.COLOR_BGR2RGB)

    # Read and process the second image
    contents2 = uploaded_file2.read()
    nparr2 = np.frombuffer(contents2, np.uint8)
    imageMat2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for display
    image_rgb2 = cv2.cvtColor(imageMat2, cv2.COLOR_BGR2RGB)

    # Display both images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb1, caption="First Image", width=300)
    with col2:
        st.image(image_rgb2, caption="Second Image", width=300)

    # Perform face similarity check
    faces1 = model.get(imageMat1)
    faces2 = model.get(imageMat2)

    if len(faces1) > 0 and len(faces2) > 0:
        embedding1 = faces1[0].embedding
        embedding2 = faces2[0].embedding
        similarity_score = float(cosine_similarity([embedding1], [embedding2])[0][0] * 100)

        st.write(f"Similarity Score: {similarity_score:.2f}%")
    else:
        st.error("No face detected in one or both images.")

