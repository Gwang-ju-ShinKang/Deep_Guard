from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow numpy opencv-python matplotlib

from tensorflow.keras.models import load_model

# 모델 경로 설정
model_path = "/content/drive/MyDrive/Colab Notebooks/데이터디자인_딥러닝/deepfake_model.h5"

# 모델 로드
model = load_model(model_path)
model.summary()

import cv2
import numpy as np

# 테스트 이미지 로드 및 전처리
test_img_path = "/content/drive/MyDrive/Colab Notebooks/데이터디자인_딥러닝/data/testfake.png"  # 테스트 이미지 경로
test_img = cv2.imread(test_img_path)
test_img = cv2.resize(test_img, (224, 224))  # 모델에 맞는 크기로 조정
test_img = test_img / 255.0  # 정규화
test_img = np.expand_dims(test_img, axis=0)  # 배치 차원 추가

print("모델 입력 크기:", model.input_shape)

# 예측 수행
prediction = model.predict(test_img)
class_idx = np.argmax(prediction, axis=1)  # 예측된 클래스 인덱스
print("예측 결과:", "딥페이크" if class_idx[0] == 0 else "정상")
print("예측 확률:", prediction)

# 결과값
# 예측 결과 : 딥페이크
# 예측 확률 : [[1.3727981e-06]]

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model_path = "/content/drive/MyDrive/Colab Notebooks/데이터디자인_딥러닝/deepfake_model.h5"  # 모델 경로
model = load_model(model_path)

# 테스트 이미지 경로
original_image_path = "/content/drive/MyDrive/Colab Notebooks/데이터디자인_딥러닝/data/real/real_1.png"  # 원본 이미지 경로
deepfake_image_path = "/content/drive/MyDrive/Colab Notebooks/데이터디자인_딥러닝/data/faketest/fake_3.png"  # 딥페이크 이미지 경로


# 이미지 전처리 함수
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)  # 모델 입력 크기에 맞춰 조정
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
    img = img / 255.0  # 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    return img

# 확률 계산 함수
def predict_deepfake(model, image_path):
    # 모델 입력 크기 확인
    input_shape = model.input_shape[1:3]  # 예: (224, 224)
    preprocessed_img = preprocess_image(image_path, target_size=input_shape)
    
    # 예측 수행
    prediction = model.predict(preprocessed_img)
    
    # 출력 해석
    if prediction.shape[1] == 2:  # Softmax (다중 클래스)
        deepfake_prob = prediction[0][1]  # 딥페이크 클래스의 확률
    else:  # Sigmoid (이진 분류)
        deepfake_prob = prediction[0][0]  # 딥페이크 확률
    
    return deepfake_prob


# 원본 이미지 예측
original_prob = predict_deepfake(model, original_image_path)
print(f"원본 이미지의 딥페이크 확률: {original_prob * 100:.2f}%")

# 딥페이크 이미지 예측
deepfake_prob = predict_deepfake(model, deepfake_image_path)
print(f"딥페이크 이미지의 딥페이크 확률: {deepfake_prob * 100:.2f}%")

# 원본 이미지의 딥페이크 확률: 0.00%
# 딥페이크 이미지의 딥페이크 확률: 0.00%