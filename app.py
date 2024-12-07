import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 로드
model = load_model('my_model.h5')

def load_and_preprocess_image(uploaded_file):
    """이미지를 로드하고 전처리하는 함수"""
    img = image.load_img(uploaded_file, target_size=(128, 128))  # 모델에 맞는 크기로 조정
    img_array = np.expand_dims(img, axis=0) / 255.  # 정규화
    return img_array

def predict_image(model, img_array):
    """이미지를 예측하는 함수"""
    prediction = model.predict(img_array)
    return prediction[0][0]  # 확률 반환

# Streamlit UI 설정
st.title("딥페이크 이미지 판별기")

# 원본 이미지 업로드
original_image_file = st.file_uploader("원본 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
# 페이크 이미지 업로드
fake_image_file = st.file_uploader("페이크 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if original_image_file and fake_image_file:
    # 이미지 전처리
    original_img_array = load_and_preprocess_image(original_image_file)
    fake_img_array = load_and_preprocess_image(fake_image_file)

    # 예측 수행
    original_prediction = predict_image(model, original_img_array)
    fake_prediction = predict_image(model, fake_img_array)

    # 결과 출력
    st.write(f"**원본 이미지 예측 (Fake 확률)**: {original_prediction:.4f}")
    st.write(f"**페이크 이미지 예측 (Fake 확률)**: {fake_prediction:.4f}")

    # 딥페이크 여부 판단
    if fake_prediction > 0.5:
        st.write("페이크 이미지는 'Fake'로 예측됩니다.")
    else:
        st.write("페이크 이미지는 'Real'로 예측됩니다.")
