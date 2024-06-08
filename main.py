import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from yolov5 import train

nama_aksara = [
    'Carik', #0
    'Carik kalih', #1
    'taling', #2
    'panti' #3
    'cecek', #4
    'pamada', #5
    'pamada', #6
    '0', #7
    '1', #8
    '2', #9
    '3', #10
    '4', #11
    '5', #12
    '6', #13
    '7', #14
    '8', #15
    '9', #16
    'a-kara', #17
    'a_kara tedong', #18
    'i-kara', #19
    'i-kara tedong', #20
    'u-kara', #21
    'u-kara tedong', #22
    'e-kara', #23
    'je-jera', #24
    'o-kara', #25
    'o-kara tedong', #26
    'Ha', #27
    'Na', #28
    'Ca', #29
    'Ra', #30
    'Ka', #31
    'Da', #32
    'Ta', #33
    'Sa', #34
    'Wa', #35
    'La', #36
    'Ma', #37
    'Ga', #38
    'Ba', #39
    'Nga', #40
    'Pa', #41
    'Ja', #42
    'Ya', #43
    'Nya', #44
    'Na Rambat', #45
    'Da Madu', #46
    'Ta Tawa', #47
    'Ta Latik', #48
    'Sa Saga', #49
    'Sa Sapa', #50
    'Ga Gora', #51
    'Ba Kembang', #52
    'Pa Kapal', #53
    'Ca Laca', #54
    'Kha', #55
    'Taleng', #56
    'Ulu', #57
    'Ulu Sari', #58
    'Suku', #59
    'Suku Ilut', #60
    'Taleng', #61
    'Taleng Marepa', #62
    'Taleng Tedong', #63
    'Taleng Marepa Tedong', #64
    'Pepet', #65
    'Pepet Tedong', #66
    'Ulu Candra', #67
    'Ulu Ricem', #68
    'Cecek', #69
    'Surang', #70
    'Bisah', #71
    'Adeg-adeg', #72
    'Gantungan A/Ha', #73
    'Gantungan Na', #74
    'Gantungan Ca', #75
    'Gantungan Ra', #76
    'Gantungan Ka', #77
    'Gantungan Da', #78
    'Gantungan Ta', #79
    'Gempelan Sa', #80
    'Gantungan Wa', #81
    'Gantungan La', #82
    'Gantungan Ma', #83
    'Gantungan Ga', #84
    'Gantungan Ba', #85
    'Gantungan Nga', #86
    'Gempelan Pa', #87
    'Gantungan Ja', #88
    'Gantungan Ya', #89
    'Gantungan Nya', #90
    'Gantungan Na Rambat', #91
    'Gantungan Da Madu', #92
    'Guung Mecelek', #93
    'Gantungan Ta Latik', #94
    'Gantungan Sa Saga', #95
    'Gempelan Sa Sapa', #96
    'Ga Gora', #97
    'Gantunga Ba Kembang', #98
    'Gantungan Ta Latik', #99
    'Gantungan Ca Laca', #100
    'Gantungan Kha', #101
]

# Load YOLO model
model_path = 'D:/Kampus/Semester 4/MBKM Dago AI and Web/Aksara Bali/Segm_Aksara_Bali/models/yolo_model.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

st.title('Aplikasi Segmentasi Aksara Bali')

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    
    # Convert image to numpy array
    image_np = np.array(image)

    # Perform inference
    results = model(image_np)

    # Draw bounding boxes and labels
    for *box, conf, cls in results.xyxy[0]:  # xyxy format
        label = nama_aksara[int(cls)]
        cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image_np, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert back to PIL Image and display
    result_image = Image.fromarray(image_np)
    st.image(result_image, caption='Hasil Segmentasi.', use_column_width=True)
