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

#####################
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# EfficientNet 모델 생성 함수
def create_efficientnet(version='B0', input_shape=None):
    if version == 'B0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif version == 'B1':
        base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
    elif version == 'B2':
        base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(260, 260, 3))
    elif version == 'B3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
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
########################
# 각 EfficientNet 모델 생성 및 학습
for version in ['B0', 'B1', 'B2', 'B3']:
    train_generator, validation_generator, test_generator = get_generators(version)
    efficientnet_model = create_efficientnet(version=version)
    efficientnet_model.fit(train_generator, validation_data=validation_generator, epochs=10)
#########################
efficientnet_model.evaluate(test_generator)
############################
efficientnet_model.trainable = True
for layer in efficientnet_model.layers[:-20]:  # 상위 20개 층만 학습
    layer.trainable = False
# 코드 해석
# base_model.trainable = True:
# base_model의 모든 층을 학습 가능하도록 설정합니다. 이 단계에서는 모든 층이 업데이트될 수 있도록 허용합니다.
##
# for layer in base_model.layers[:-20]::
# base_model의 모든 층 중에서 마지막 20개 층을 제외한 나머지 층을 반복(iterate)합니다. base_model.layers는 모델의 모든 층을 포함하는 리스트입니다.
# [:-20]는 리스트 슬라이싱을 사용하여 마지막 20개 층을 제외한 부분을 선택합니다.
##
# layer.trainable = False:
# 반복문을 통해 선택된 각 층의 trainable 속성을 False로 설정합니다. 이는 해당 층이 학습 중에 가중치가 업데이트되지 않도록 고정하는 것입니다.
##
# 전체적인 의미
# 이 코드는 전이 학습을 수행할 때 유용합니다. 일반적으로 사전 학습된 모델의 하위 층(특징 추출기)은 이미 유용한 특징을 학습했기 때문에, 이 층들은 고정(freeze)하여 학습하지 않고 상위 층(주로 분류기에 해당하는 층)만 학습하도록 설정합니다.
# 상위 20개 층: 모델의 마지막 20개 층은 새로운 데이터셋에 맞게 조정하기 위해 학습할 수 있습니다.
# 하위 층 고정: 이미 학습된 특징을 활용하여 과적합(overfitting)을 방지하고 학습 속도를 높이는 데 기여합니다.
# 이러한 방식으로 모델을 조정하면, 전이 학습의 이점을 극대화할 수 있습니다. 추가적인 질문이 있으면 언제든지 말씀해 주세요!   
##############
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
efficientnet_model.fit(train_generator, validation_data=validation_generator, epochs=30, callbacks=[lr_scheduler])
##################
