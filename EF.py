from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        'data/data/Dataset/Train',
        target_size=target_size,
        batch_size=32,
        class_mode='binary'
    )
    validation_generator = val_test_datagen.flow_from_directory(
        'data/data/Dataset/Validation',
        target_size=target_size,
        batch_size=32,
        class_mode='binary'
    )
    test_generator = val_test_datagen.flow_from_directory(
        'data/data/Dataset/Test',
        target_size=target_size,
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    return train_generator, validation_generator, test_generator
#####
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# EfficientNet 모델 생성 함수
def create_efficientnet(version='B0', weights_path=None, input_shape=None):
    if version == 'B0':
        base_model = EfficientNetB0(weights=weights_path if weights_path else 'imagenet', include_top=False, input_shape=(224, 224, 3))
    elif version == 'B1':
        base_model = EfficientNetB1(weights=weights_path if weights_path else 'imagenet', include_top=False, input_shape=(240, 240, 3))
    elif version == 'B2':
        base_model = EfficientNetB2(weights=weights_path if weights_path else 'imagenet', include_top=False, input_shape=(260, 260, 3))
    elif version == 'B3':
        base_model = EfficientNetB3(weights=weights_path if weights_path else 'imagenet', include_top=False, input_shape=(300, 300, 3))
    else:
        raise ValueError("Invalid version. Choose from 'B0', 'B1', 'B2', 'B3'.")

    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 각 EfficientNet 모델 생성 및 학습
for version in ['B0', 'B1', 'B2', 'B3']:
    weights_path = f'data/efficientnet{version.lower()}_notop.h5'  # 로컬 가중치 파일 경로
    train_generator, validation_generator, test_generator = get_generators(version)
    efficientnet_model = create_efficientnet(version=version, weights_path=weights_path)
    efficientnet_model.fit(train_generator, validation_data=validation_generator, epochs=10)

#########
efficientnet_model.evaluate(test_generator)
#######
efficientnet_model.trainable = True
for layer in efficientnet_model.layers[:-20]:  # 상위 20개 층만 학습
    layer.trainable = False
    #######
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    if epoch < 10:
        return 1e-3
    elif epoch < 20:
        return 1e-4
    else:
        return 1e-5

lr_scheduler = LearningRateScheduler(lr_schedule)
efficientnet_model.fit(train_generator, validation_data=validation_generator, 
                       epochs=30, callbacks=[lr_scheduler])
############
model.save('EF_model.h5')