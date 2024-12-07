from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 전처리 및 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,            # 회전 범위
    width_shift_range=0.2,        # 가로 이동 범위
    height_shift_range=0.2,       # 세로 이동 범위
    shear_range=0.2,              # 전단 변환
    zoom_range=0.2,               # 확대/축소 범위
    horizontal_flip=True           # 수평 뒤집기
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    'data/data/Dataset/Train',           # 학습 데이터 경로
    target_size=(224, 224),              # 이미지 크기 조정
    batch_size=32,
    class_mode='binary',                 # 이진 분류
    shuffle=True                         # 학습 데이터는 섞기
)

validation_generator = val_test_datagen.flow_from_directory(
    'data/data/Dataset/Validation',      # 검증 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False                         # 검증 데이터는 섞지 않음
)

test_generator = val_test_datagen.flow_from_directory(
    'data/data/Dataset/Test',            # 테스트 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False                         # 테스트 데이터는 섞지 않음
)

# 데이터 생성자 확인
print(f'Train samples: {train_generator.samples}')
print(f'Validation samples: {validation_generator.samples}')
print(f'Test samples: {test_generator.samples}')
######
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape=(224, 224, 3)):
    model = Sequential([
        # Convolutional Layers
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten and Fully Connected Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

########
  history = model.fit(
      train_generator,
      validation_data=validation_generator,
      epochs=20,  # 학습 반복 횟수
      steps_per_epoch=train_generator.samples // train_generator.batch_size,
      validation_steps=validation_generator.samples // validation_generator.batch_size
  )
########
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    'test_data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 모델 평가
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

#######
import matplotlib.pyplot as plt

# 학습 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()
########
model.save('cnn_model.h5')