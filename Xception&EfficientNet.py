# Xception&EfficientNet combine model

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
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/test/dataset/Dataset/Train/',           # 학습 데이터 경로
    target_size=(128, 128),     # 이미지 크기 조정
    batch_size=32,
    class_mode='binary'         # 이진 분류
)

validation_generator = val_test_datagen.flow_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/test/dataset/Dataset/Validation/',      # 검증 데이터 경로
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/test/dataset/Dataset/Test/',            # 테스트 데이터 경로
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False               # 테스트 데이터는 섞지 않음 (정확도 평가를 위해)
)
########################
from tensorflow.keras.applications import EfficientNetB0, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def create_efficientnet(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_xception(input_shape=(224, 224, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
#############################
# EfficientNet 학습
efficientnet = create_efficientnet()
efficientnet.fit(train_generator, validation_data=validation_generator, epochs=10)

# Xception 학습
xception = create_xception()
xception.fit(train_generator, validation_data=validation_generator, epochs=10)
##############################
import numpy as np

def ensemble_predict(models, img_array):
    predictions = [model.predict(img_array) for model in models]
    avg_prediction = np.mean(predictions, axis=0)  # 평균값 계산
    return avg_prediction

# 예측 수행
models = [efficientnet, xception]
img_array = np.expand_dims(image.load_img('test.jpg', target_size=(224, 224)), axis=0) / 255.
prediction = ensemble_predict(models, img_array)
print("Fake" if prediction > 0.5 else "Real")
################################
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

def create_hybrid_model(input_shape=(224, 224, 3)):
    # EfficientNet
    efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    efficientnet_base.trainable = False
    efficientnet_features = GlobalAveragePooling2D()(efficientnet_base.output)
    
    # Xception
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    xception_base.trainable = False
    xception_features = GlobalAveragePooling2D()(xception_base.output)
    
    # Feature Concatenation
    combined_features = Concatenate()([efficientnet_features, xception_features])
    
    # Fully Connected Layer
    x = Dense(256, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[efficientnet_base.input, xception_base.input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
#######################################

# 모델 학습
hybrid_model = create_hybrid_model()
hybrid_model.fit(train_generator, validation_data=validation_generator, epochs=10)

##########################################
# 예측
img_array = np.expand_dims(image.load_img('test.jpg', target_size=(224, 224)), axis=0) / 255.
prediction = hybrid_model.predict(img_array)
print("Fake" if prediction > 0.5 else "Real")
