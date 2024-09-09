from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r', encoding='utf-8').readlines()

st.header('🍇포도 판별기🍇')
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
# 들어온 이미지를 224 x 224 x 3차원으로 변환하기 위해서 빈 벡터를 만들어 놓음
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img_file_buffer = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if img_file_buffer is not None:
    # # To read image file buffer as a PIL Image:
    # image = Image.open(img_file_buffer) # 입력받은 사진을 행렬로 변환

    # # To convert PIL Image to numpy array:
    # img_array = np.array(image) # ndarray로 변환

    # Replace this with the path to your image
    # 원본 이미지 불러오기
    image = Image.open(img_file_buffer).convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    # 모델에 들어갈 수 있는 224 x 224 사이즈로 변환 
    # 보간 방식 : Image.Resampling.LANCZOS 
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    # 이미지를 넘파이 행렬로 변환 
    image_array = np.asarray(image)

    # Normalize the image
    # 모델이 학습했을 때 Nomalize 한 방식대로 이미지를 Nomalize 
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    # 빈 ARRAY에 전처리를 완료한 이미지를 복사
    data[0] = normalized_image_array

    # run the inference
    # h5 모델에 예측 의뢰 
    prediction = model.predict(data)
    # 높은 신뢰도가 나온 인덱의 인덱스 자리를 저장
    index = np.argmax(prediction)

    # labels.txt 파일에서 가져온 값을 index로 호출
    # 좋아하는 만화 선택하세요 - 만화 제목(text 리스트)랑 img 경로 리스트 일치 시킬 때 인덱스 활용한 것과 같은 방법
    class_name = class_names[index]

    # 예측 결과에서 신뢰도를 꺼내 옵니다  
    confidence_score = prediction[0][index]

    st.write('Class:', class_name[2:], end="")
    st.write('Confidence score:', confidence_score)