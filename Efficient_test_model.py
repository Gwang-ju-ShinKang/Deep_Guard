# 4중 EfficientNet Model

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
##############
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def create_efficientnet_b0(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Transfer Learning
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
#####################
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def create_efficientnet_b0(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Transfer Learning
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#####################
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def create_efficientnet_b0(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Transfer Learning
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
###########################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('efficientnet_b0.h5', save_best_only=True, monitor='val_loss')
]
model_b0 = create_efficientnet_b0()
history_b0 = model_b0.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks
)
#############################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('efficientnet_b0.h5', save_best_only=True, monitor='val_loss')
]
model_b1 = create_efficientnet_b0()
history_b1 = model_b1.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks
)
##############################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('efficientnet_b0.h5', save_best_only=True, monitor='val_loss')
]
model_b2 = create_efficientnet_b0()
history_b2 = model_b2.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks
)
#############################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('efficientnet_b0.h5', save_best_only=True, monitor='val_loss')
]
model_b3 = create_efficientnet_b0()
history_b3 = model_b3.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks
)
#################################
from tensorflow.keras.models import load_model

model_b0 = load_model('efficientnet_b0.h5')
model_b1 = load_model('efficientnet_b1.h5')
model_b2 = load_model('efficientnet_b2.h5')
model_b3 = load_model('efficientnet_b3.h5')
#################################
import numpy as np

def ensemble_predict(models, img_array):
    predictions = [model.predict(img_array) for model in models]
    avg_prediction = np.mean(predictions, axis=0)  # 평균값 계산
    return avg_prediction

# 예측 수행
models = [model_b0, model_b1, model_b2]
img_array = np.expand_dims(image.load_img('test.jpg', target_size=(224, 224)), axis=0) / 255.
prediction = ensemble_predict(models, img_array)
print("Fake" if prediction > 0.5 else "Real")




