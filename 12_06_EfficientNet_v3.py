from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# 데이터 증강 및 전처리
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

# 데이터 생성 함수
def get_generators(version):
    input_sizes = {
        'B0': (224, 224),
        'B1': (240, 240),
        'B2': (260, 260),
        'B3': (300, 300)
    }
    target_size = input_sizes[version]
    
    train_generator = train_datagen.flow_from_directory(
        'data/split_data/train',
        target_size=target_size,
        batch_size=32,
        class_mode='binary'
    )
    validation_generator = val_test_datagen.flow_from_directory(
        'data/split_data/validation',
        target_size=target_size,
        batch_size=32,
        class_mode='binary'
    )
    test_generator = val_test_datagen.flow_from_directory(
        'data/split_data/test',
        target_size=target_size,
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    return train_generator, validation_generator, test_generator

# EfficientNet 모델 생성 함수
def create_efficientnet(version='B0', input_shape=(224, 224, 3)):
    if version == 'B0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif version == 'B1':
        base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
    elif version == 'B2':
        base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(260, 260, 3))
    elif version == 'B3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    else:
        raise ValueError("Invalid version. Choose from 'B0', 'B1', 'B2', 'B3'.")

    # Fine-tuning: 상위 층 동결
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # 마지막 20개 층만 학습
        layer.trainable = False

    # 모델 구성
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)  # Batch Normalization 추가
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)  # L2 Regularization 추가
    x = Dropout(0.5)(x)  # Dropout 비율 조정
    output = Dense(1, activation='sigmoid')(x)  # Binary Classification

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Early Stopping 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 각 EfficientNet 모델 생성 및 학습
for version in ['B0', 'B1', 'B2', 'B3']:
    train_generator, validation_generator, test_generator = get_generators(version)
    efficientnet_model = create_efficientnet(version=version)
    
    # Learning Rate Scheduler 설정
    def lr_schedule(epoch):
        if epoch < 10:
            return 1e-3
        elif epoch < 20:
            return 1e-4
        else:
            return 1e-5

    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # 모델 학습
    efficientnet_model.fit(train_generator, validation_data=validation_generator, epochs=30, 
                            callbacks=[early_stopping, lr_scheduler])

# 모델 평가
evaluation_results = efficientnet_model.evaluate(test_generator)
print("Test Loss and Accuracy:", evaluation_results)
