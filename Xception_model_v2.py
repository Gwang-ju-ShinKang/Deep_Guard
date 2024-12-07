import zipfile
import os
import shutil

# ZIP 파일 압축 해제
def extract_zip(zip_path, extract_path):
    """
    ZIP 파일을 추출합니다.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Dataset extracted to {extract_path}")

# 실행 경로 설정
zip_path = "data/data.zip"
extract_path = "data/data"

# 실행
extract_zip(zip_path, extract_path)

###########
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 전처리 및 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    fill_mode='nearest'  # 변형 후 비어있는 부분을 채우는 방법
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    'data/data/Dataset/Train',           # 학습 데이터 경로
    target_size=(224, 224),     # 이미지 크기 조정
    batch_size=32,
    class_mode='binary'         # 이진 분류
)

validation_generator = val_test_datagen.flow_from_directory(
    'data/data/Dataset/validation',      # 검증 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    'data/data/Dataset/test',            # 테스트 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False               # 테스트 데이터는 섞지 않음 (정확도 평가를 위해)
)
####################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import Xception
##################
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model(input_shape=(224, 224, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Fine-tuning: Last few layers of base_model를 학습 가능하게 설정
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
##############
# 모델 생성
model = create_model()
#############
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# 모델 학습
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[early_stopping, checkpoint]
)
###############
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")
################
from tensorflow.keras.models import load_model

# 최적의 모델 로드
best_model = load_model('best_model.h5')

# 모델 평가
evaluation_results = best_model.evaluate(test_generator)
print("Best Model Test Loss and Accuracy:", evaluation_results)
###############
# 시각화 
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
#################
from tensorflow.keras.preprocessing import image
import numpy as np
# image: Keras의 image 모듈을 사용하여 이미지를 로드하고 처리합니다.
# numpy: 배열을 다루기 위한 NumPy 라이브러리를 사용합니다.

def predict_image(model, img_path):
# predict_image라는 함수를 정의하며, 두 개의 인자를 받습니다: 
# model (예측할 모델)과 img_path (예측할 이미지의 경로).
    img = image.load_img(img_path, target_size=(224, 224))
# image.load_img(img_path, target_size=(128, 128)): 지정된 경로(img_path)에서 이미지를 로드하고, 
# 크기를 (128, 128)로 조정합니다. 이 크기는 모델이 기대하는 입력 크기여야 합니다.
    img_array = np.expand_dims(img, axis=0) / 255. # / 255.: 이미지를 0과 1 사이의 값으로 정규화
# np.expand_dims(img, axis=0): 이미지를 모델에 입력하기 위해 차원을 추가합니다. 
# 원래 이미지는 (height, width, channels) 형태이지만, 
# 모델 입력은 (batch_size, height, width, channels) 형태여야 하므로 첫 번째 축을 추가합니다.
    prediction = model.predict(img_array)
# model.predict(img_array): 전처리된 이미지를 모델에 입력하여 예측을 수행합니다. 
# 이 함수는 예측된 확률 값을 반환합니다.
    return "Fake" if prediction[0][0] > 0.5 else "Real"
# prediction[0][0]: 모델의 예측 결과에서 첫 번째 샘플의 첫 번째 값을 가져옵니다. 
# 이는 이진 분류 문제에서 "Fake" 또는 "Real"을 나타내는 확률 값입니다.
# return "Fake" if prediction[0][0] > 0.5 else "Real": 
# 예측된 확률이 0.5보다 크면 "Fake"로, 그렇지 않으면 "Real"로 반환합니다. 이는 이진 분류의 기준입니다.
##############
# 모델 저장
model.save('Xc_model.h5')
###############
from tensorflow.keras.models import load_model

# 저장된 모델 로드
loaded_model = load_model('Xc_model.h5')

loaded_model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 모델 평가
evaluation_results = loaded_model.evaluate(test_generator)
print("Loaded Model Test Loss and Accuracy:", evaluation_results)