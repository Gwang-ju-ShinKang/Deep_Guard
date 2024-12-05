# Xception model test
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

#################
from tensorflow.keras.applications import Xception

def create_model(input_shape=(128, 128, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Transfer Learning
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary Classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
###############
# 하드웨어 사용
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
##################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[early_stopping, checkpoint]
)
######################
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

#########################
# 시각화 
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
##########################
base_model.trainable = True
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)

###########################
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = np.expand_dims(img, axis=0) / 255.
    prediction = model.predict(img_array)
    return "Fake" if prediction[0][0] > 0.5 else "Real"
