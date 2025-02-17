import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


con = list(mp_face_mesh.FACEMESH_CONTOURS)


c = []

for i in con:
    for j in i:
        c.append(j)

c = sorted(list(set(c)))
print(len(c))