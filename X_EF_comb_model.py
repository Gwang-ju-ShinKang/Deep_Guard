from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Xception 모델 생성
def create_xception(input_shape=(224, 224, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(224, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=output)

# EfficientNet 모델 생성
def create_efficientnet(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(224, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=output)
##############
# 데이터 전처리 및 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    'data/split_data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
validation_generator = val_test_datagen.flow_from_directory(
    'data/split_data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
test_generator = val_test_datagen.flow_from_directory(
    'data/split_data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
###################
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

# 모델 생성
xception = create_xception()
efficientnet = create_efficientnet()

# 개별 모델 학습
xception.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
efficientnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

xception.fit(train_generator, validation_data=validation_generator, epochs=10)
efficientnet.fit(train_generator, validation_data=validation_generator, epochs=10)

# 앙상블 모델 생성
xception_output = xception.output
efficientnet_output = efficientnet.output

# 두 모델 출력 결합
merged = concatenate([xception_output, efficientnet_output])
merged = Dense(224, activation='relu')(merged)
merged = Dropout(0.5)(merged)
final_output = Dense(1, activation='sigmoid')(merged)

# 최종 앙상블 모델 정의
ensemble_model = Model(inputs=[xception.input, efficientnet.input], outputs=final_output)

# 앙상블 모델 컴파일
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("앙상블 모델이 성공적으로 컴파일되었습니다!")
#################
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# 원본 및 테스트 이미지 로드
original_image_path = 'path_to_original_image.jpg'
test_image_path = 'path_to_test_image.jpg'

original_image = preprocess_image(original_image_path)
test_image = preprocess_image(test_image_path)

# 예측
prediction = ensemble_model.predict([original_image, test_image])
fake_probability = prediction[0][0] * 100  # 딥페이크 확률
print(f"테스트 이미지는 딥페이크일 확률이 {fake_probability:.2f}%입니다.")
##################
# 모델 저장
model.save('X_EF_model.h5')
###############
from tensorflow.keras.models import load_model

# 저장된 모델 로드
loaded_model = load_model('X_EF_model.h5')

loaded_model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 모델 평가
evaluation_results = loaded_model.evaluate(test_generator)
print("Loaded Model Test Loss and Accuracy:", evaluation_results)