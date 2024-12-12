from fastapi import FastAPI, File, UploadFile, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model
from PIL import Image
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tensorflow as tf
import io
import os

app = FastAPI()

# CORS 설정 (모든 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 의존성 주입: 데이터베이스 세션 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API 엔드포인트: 모든 데이터 가져오기
@app.get("/items")
def read_items(db: Session = Depends(get_db)):
    items = db.query(UserInfo).all()
    return items

#import os
import tensorflow as tf

# 프로젝트의 루트 경로를 base_dir로 설정 (현재 파일의 상위 디렉터리로 이동)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 현재 파일의 상위 디렉터리로 이동
model_path = os.path.join(base_dir, 'Model', 'Xception_model_V1.h5')  # Model 폴더로 경로 생성
print(f"🔍 모델 경로: {model_path}")  # 디버깅을 위해 경로 확인

try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ 모델이 성공적으로 로드되었습니다! (입력 형태: {model.input_shape})")
except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_path}")
except Exception as e:
    print(f"❌ 모델 로드 중 알 수 없는 오류가 발생했습니다: {e}")
    model = None  # 모델 로드 실패시 None으로 설정


def preprocess_image(image: Image.Image) -> np.ndarray:
    """ 이미지를 224x224x3 형태로 변환하는 함수 """
    try:
        image = image.resize((128, 128))  
        if image.mode == 'L': 
            image = image.convert('RGB')
        if image.mode == 'RGBA':  
            image = image.convert('RGB')

        image_array = np.array(image)  
        if image_array.ndim == 2:  
            image_array = np.stack([image_array] * 3, axis=-1)

        print(f"🖼️ 이미지 배열 크기: {image_array.shape}")
        image_array = image_array / 255.0  
        image_array = np.expand_dims(image_array, axis=0)  
        return image_array

    except Exception as e:
        print(f"❌ 이미지 전처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="이미지 전처리 중 오류가 발생했습니다.")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

        image = Image.open(io.BytesIO(contents))  
        image_array = preprocess_image(image)  

        try:
            if model is None:
                raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
            
            print(f"🖼️ 예측에 사용된 입력 데이터 크기: {image_array.shape}")
            prediction = model.predict(image_array)
            print(f"✅ 예측 결과: {prediction}")  

        except Exception as e:
            print(f"❌ TensorFlow 예측 오류: {e}")
            raise HTTPException(status_code=500, detail="예측 중 오류가 발생했습니다.")

        result = {"status": "success", "data": prediction.tolist()}
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ 서버 오류 발생: {e}")
        return JSONResponse(content={"error": f"서버 오류: {str(e)}"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
