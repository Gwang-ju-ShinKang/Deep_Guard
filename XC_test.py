import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

# 1. 저장된 모델 로드
model = load_model('X_EF_model.h5')

# 2. 이미지 전처리 함수 정의
def preprocess_image(image_path, target_size=(224, 224)):
    """
    이미지를 모델 입력 크기에 맞게 전처리합니다.
    """
    img = load_img(image_path, target_size=target_size)  # 이미지를 로드하고 크기 변경
    img_array = img_to_array(img) / 255.0               # 배열로 변환하고 정규화
    img_array = np.expand_dims(img_array, axis=0)       # 배치를 추가
    return img_array

# 3. 사용자로부터 이미지 선택
def select_images():
    """
    두 개의 이미지를 선택합니다: 원본 사진과 딥페이크 의심 사진.
    """
    Tk().withdraw()  # Tkinter 기본 창 숨기기
    print("원본 사진을 선택하세요:")
    original_image_path = filedialog.askopenfilename()
    print("딥페이크 의심 사진을 선택하세요:")
    suspect_image_path = filedialog.askopenfilename()
    return original_image_path, suspect_image_path

# 4. 이미지 예측 및 결과 출력
def predict_fake_probability(original_path, suspect_path):
    """
    두 이미지를 비교하여 딥페이크 확률을 계산합니다.
    """
    # 원본 및 의심 사진 전처리
    original_image = preprocess_image(original_path)
    suspect_image = preprocess_image(suspect_path)
    
    # 딥페이크 의심 사진 예측
    suspect_prediction = model.predict(suspect_image)[0][0]  # 딥페이크 확률 (0~1)
    
    # 결과 출력
    print(f"딥페이크 의심 사진이 딥페이크일 확률: {suspect_prediction * 100:.2f}%")
    
    # 원본 및 의심 사진 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(load_img(original_path))
    axes[0].set_title("원본 사진")
    axes[0].axis("off")
    axes[1].imshow(load_img(suspect_path))
    axes[1].set_title("딥페이크 의심 사진")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# 5. 실행
if __name__ == "__main__":
    original_image_path, suspect_image_path = select_images()  # 이미지 선택
    predict_fake_probability(original_image_path, suspect_image_path)  # 결과 예측
