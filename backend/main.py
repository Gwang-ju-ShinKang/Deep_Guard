from datetime import datetime, timezone, timedelta
from database import SessionLocal, engine
from fastapi import FastAPI, Depends, UploadFile, File, Form,HTTPException, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from itsdangerous import URLSafeTimedSerializer
from models import Base, UploadInfo, SessionInfo
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from keras.layers import TFSMLayer
from typing import List
import io
import json
import numpy as np
import os
import tensorflow as tf
import uuid
import uvicorn
import tensorflow as tf
import traceback

# 데이터베이스 초기화
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (보안을 위해 특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 의존성 주입: 데이터베이스 세션 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API 엔드포인트: 모든 데이터 가져오기
@app.get("/image")
def read_items(db: Session = Depends(get_db)):
    items = db.query(UploadInfo).all()
    return items

# Secret Key와 Serializer 설정
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
serializer = URLSafeTimedSerializer(SECRET_KEY)

@app.get("/create-session")
def create_session(response: Response):
    # 세션 생성
    session_data = {
        "session_id": str(uuid.uuid4()),                # 고유 세션 ID
        "created_at": datetime.now(timezone.utc).isoformat(),  # datetime -> 문자열 (ISO 8601 형식)
        "last_activity": datetime.now(timezone.utc).isoformat(),  # 마지막 활동 시간
        "session_expire_dt": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()  # 1시간 후 종료
    }
    session_id = serializer.dumps(session_data)  # 세션 데이터를 직렬화

    response.set_cookie(
        key="session_id",
        value=session_id,  # 세션 ID 값
        httponly=True,
        secure=False,  # HTTPS 환경에서는 True로 설정
        samesite="Lax"
    )

    response.set_cookie(
        key="created_at",
        value=session_data["created_at"],
        httponly=True,
        secure=False,  # HTTPS 환경에서는 True로 설정
        samesite="Lax"
    )

    return {"message": "세션 생성 완료", "session_data": session_data}

@app.get("/get-session", response_class=JSONResponse)
def get_session(request: Request):
    # 쿠키에서 세션 ID 가져오기
    session_id = request.cookies.get("session_id")

    if not session_id:
        # 세션 ID가 없는 경우
        return JSONResponse(status_code=400, content={"message": "세션 없음"})

    try:
        # 세션 ID를 디코딩하여 확인
        session_data = serializer.loads(session_id, max_age=3600)  # 세션 유효 시간: 1시간
        return {"message": "세션 데이터 확인", "data": session_data}
    except Exception as e:
        # 예외 처리 및 디버깅 메시지 반환
        return JSONResponse(status_code=400, content={"message": f"세션 오류: {str(e)}"})
    
# 디바이스 정보
# Pydantic 모델 정의
class DeviceInfo(BaseModel):
    userAgent: str
    platform: str
    language: str


# GET 요청에 대한 처리 추가
@app.get("/device-info/")
async def get_device_info():
    return {"message": "This endpoint only accepts POST requests for sending data."}

@app.post("/device-info/")
async def receive_device_info(
    device_info: DeviceInfo,
    db: Session = Depends(get_db),
    session_id: str = Cookie(None),
    created_at: str = Cookie(None)
):
    # 수신된 장치 정보 로그
    print("Received Device Info:", device_info.dict())
    print("Session ID:", session_id)

    # 세션 ID가 없으면 오류 반환
    if not session_id:
        return {"message": "Session ID is required."}

    # 세션 데이터를 복원
    session_data = serializer.loads(session_id)
    
    # 현재 시간을 마지막 활동 시간과 세션 종료 시간으로 업데이트
    session_data["create_session"] = datetime.now(timezone.utc)
    session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
    session_data["session_expire_dt"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    # created_at을 datetime 객체로 변환
    log_time = datetime.fromisoformat(created_at) if created_at else None

    # 디바이스 정보를 데이터베이스에 저장
    try:
        new_device_info = SessionInfo(
            log_device=device_info.userAgent,
            session_id=session_id,  # create_session에서 받은 session_id
            session_created_at=session_data["create_session"],  # datetime 객체로 변환된 log_time
            session_active_duration=None,  # 나중에 계산하여 업데이트 필요
            session_expire_dt=session_data["session_expire_dt"]
        )
        
        print("Prepared data for DB:", new_device_info)
        db.add(new_device_info)  # 인스턴스를 추가
        db.commit()              # 변경 사항 저장
        db.refresh(new_device_info)  # 새로 추가된 데이터 반환
        
        return {"message": "Device information saved", "data": new_device_info.session_idx}  # session_idx 반환
    except Exception as e:
        print("Error saving to DB:", e)
        return {"message": "Error saving device information", "error": str(e)}  # 오류 메시지 추가

""" # 모델 로드
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 현재 파일의 상위 디렉터리로 이동
model_path = os.path.join(base_dir, 'Model', 'Xception_model.h5')  # Model 폴더로 경로 생성
print(f"🔍 모델 경로: {model_path}")  # 디버깅을 위해 경로 확인

try:
    # model = tf.keras.models.load_model(model_path)
    # 🛠️ custom_objects 추가
    model = load_model(
        model_path, 
        custom_objects={'BatchNormalization': BatchNormalization}
    )
    print(f"✅ 모델이 성공적으로 로드되었습니다! (입력 형태: {model.input_shape})")
except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_path}")
except Exception as e:
    print(f"❌ 모델 로드 중 알 수 없는 오류가 발생했습니다: {e}")
    model = None  # 모델 로드 실패시 None으로 설정 """

# 📁 모델 저장 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 현재 파일의 상위 디렉터리로 이동
model_dir = os.path.join(base_dir, 'Model', 'Xception_model')  # 디렉터리로 경로 생성

# 🛠️ 모델 로드
try:
    # SavedModel 형식으로 모델을 로드합니다.
    model = tf.keras.models.load_model(model_dir)
    print(f"✅ 모델이 성공적으로 로드되었습니다! (입력 형태: {model.input_shape})")
except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_dir}")
except Exception as e:
    print(f"❌ 모델 로드 중 알 수 없는 오류가 발생했습니다: {e}")
    model = None  # 모델 로드 실패시 None으로 설정
    
# 이미지 전처리 
def preprocess_image(image: Image.Image) -> np.ndarray:
    """ 이미지를 224x224x3 형태로 변환하는 함수 """
    try:
        image = image.resize((224, 224))  
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

# 이미지 업로드
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

        image = Image.open(io.BytesIO(contents))  
        image_array = preprocess_image(image)  # 이미지를 전처리하여 224x224로 변환

        try:
            if model is None:
                raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
            
            print(f"🖼️ 예측에 사용된 입력 데이터 크기: {image_array.shape}")
            
           #두 개의 입력을 동시에 모델에 전달
           #prediction = model.predict([image_array, image_array])  # (image_array, image_array)로 전달
           #print(f"✅ 예측 결과: {prediction}")   

            # 다양한 입력 아닌 해당 이미지 하나를 다른 것에 복제해서 반복 전달하지 않\uc도록 수정
            prediction = model.predict(image_array)  # 하나의 입력만 전달
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
 
