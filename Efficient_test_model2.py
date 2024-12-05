# 4중 EfficientNet Model

import zipfile
import os
import shutil
from sklearn.model_selection import train_test_split

# ZIP 파일 압축 해제
def extract_zip(zip_path, extract_path):
    """
    ZIP 파일을 추출합니다.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Dataset extracted to {extract_path}")

# 데이터 분할 함수
def split_data(source_dir, output_dir, train_size=0.7, val_size=0.15):
    """
    데이터를 train, validation, test로 분할합니다.
    """
    if train_size + val_size >= 1.0:
        raise ValueError("train_size + val_size must be less than 1.0")
    
    categories = ['fake', 'real']
    for category in categories:
        category_dir = os.path.join(source_dir, category)
        if not os.path.exists(category_dir):
            raise FileNotFoundError(f"Category folder not found: {category_dir}")
        
        files = [os.path.join(category_dir, file) for file in os.listdir(category_dir) 
                 if os.path.isfile(os.path.join(category_dir, file))]
        
        # Train/Validation/Test 분할
        train_files, test_files = train_test_split(files, test_size=1 - train_size, random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=(1 - train_size - val_size) / (1 - train_size), random_state=42)
        
        # 데이터 저장
        for split, split_files in zip(['train', 'validation', 'test'], [train_files, val_files, test_files]):
            split_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for file in split_files:
                shutil.copy(file, split_dir)
    print("Data successfully split into train, validation, and test sets.")

# 실행 경로 설정
zip_path = "/content/drive/MyDrive/Colab Notebooks/test/dataset/Dataset.zip"
extract_path = "/content/drive/MyDrive/Colab Notebooks/test/dataset"
output_dir = "/content/drive/MyDrive/Colab Notebooks/test/split_data"

# 실행
extract_zip(zip_path, extract_path)
split_data(extract_path, output_dir)

##################
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
        '/content/drive/MyDrive/Colab Notebooks/test/dataset/Dataset/Train',
        target_size=target_size,
        batch_size=32,
        class_mode='binary'
    )
    validation_generator = val_test_datagen.flow_from_directory(
        '/content/drive/MyDrive/Colab Notebooks/test/dataset/Dataset/Validation',
        target_size=target_size,
        batch_size=32,
        class_mode='binary'
    )
    test_generator = val_test_datagen.flow_from_directory(
        '/content/drive/MyDrive/Colab Notebooks/test/dataset/Dataset/Test',
        target_size=target_size,
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    return train_generator, validation_generator, test_generator

##############
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

#####################
print(len(train_generator))  # train_generator에 포함된 배치 수 확인
print(len(validation_generator))  # validation_generator에 포함된 배치 수 확인
####################

# 각 EfficientNet 모델 생성 및 학습
for version in ['B0', 'B1', 'B2', 'B3']:
    train_generator, validation_generator, test_generator = get_generators(version)
    efficientnet_model = create_efficientnet(version=version)
    efficientnet_model.fit(train_generator, validation_data=validation_generator, epochs=10)

#####################
base_model.trainable = True
for layer in base_model.layers[:-20]:  # 상위 20개 층만 학습
    layer.trainable = False

######################
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
model.fit(train_generator, validation_data=validation_generator, epochs=30, callbacks=[lr_scheduler])
########################
model.save('custom_deepfake_model.h5')

