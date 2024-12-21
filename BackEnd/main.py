from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from datetime import datetime, timezone, timedelta
from models import Base, UploadInfo, SessionInfo, DeviceInfo,ImageBackupInfo
from itsdangerous import URLSafeTimedSerializer
from tensorflow.keras.models import model_from_json
from PIL import Image,UnidentifiedImageError
from pydantic import BaseModel
import numpy as np
import os
import uuid
import io
import tensorflow as tf
import base64
import json

# 데이터베이스 초기화
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 설정
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

# Secret Key와 Serializer 설정
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
serializer = URLSafeTimedSerializer(SECRET_KEY)

# 📁 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 상위 디렉터리로 이동
model_json_path = os.path.join(base_dir, 'BackEnd', 'Models', 'model.json')  # 모델 구조 경로
model_weights_path = os.path.join(base_dir, 'BackEnd', 'Models', 'model_weights.h5')  # 가중치 파일 경로

print(f"🔍 모델 JSON 경로: {model_json_path}")
print(f"🔍 모델 가중치 경로: {model_weights_path}")

# 파일 경로 확인 (디버깅용)
if not os.path.isfile(model_json_path):
    print(f"❌ 모델 JSON 파일을 찾을 수 없습니다: {model_json_path}")
if not os.path.isfile(model_weights_path):
    print(f"❌ 모델 가중치 파일을 찾을 수 없습니다: {model_weights_path}")

try:
    # 📘 모델 불러오기 단계
    with open(model_json_path, 'r') as f:
        model_config = json.load(f)

    # 🔥 SeparableConv2D의 불필요한 인자 삭제
    for layer in model_config['config']['layers']:
        if layer['class_name'] == 'SeparableConv2D':
            layer['config'].pop('groups', None)
            layer['config'].pop('kernel_initializer', None)
            layer['config'].pop('kernel_regularizer', None)
            layer['config'].pop('kernel_constraint', None)
            
        # ✅ BatchNormalization의 axis 수정 (리스트 → 단일 정수)
        if layer['class_name'] == 'BatchNormalization' and 'axis' in layer['config']:
            axis_value = layer['config']['axis']
            if isinstance(axis_value, list) and len(axis_value) == 1:
                layer['config']['axis'] = axis_value[0]  # ✅ 리스트에서 단일 값으로 변환

    # 🔥 custom_objects에 필요한 클래스 추가
    custom_objects = {'Functional': tf.keras.Model, 'BatchNormalization': tf.keras.layers.BatchNormalization}

    # 🔥 모델 다시 생성
    model = model_from_json(json.dumps(model_config), custom_objects=custom_objects)
    model.load_weights(model_weights_path)
    print(f"✅ 모델이 성공적으로 로드되었습니다! (입력 형태: {model.input_shape})")

except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_json_path} 또는 {model_weights_path}")
    model = None
except Exception as e:
    print(f"❌ 모델 로드 중 알 수 없는 오류가 발생했습니다: {e}")
    model = None
# 🖼️ 이미지 전처리 함수
def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        image = image.resize((299, 299))
        if image.mode in ['L', 'RGBA']:
            image = image.convert('RGB')

        image_array = np.array(image)
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)

        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    except Exception as e:
        print(f"❌ 이미지 전처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="이미지 전처리 중 오류가 발생했습니다.")

# 🚀 FastAPI 앱 생성 및 엔드포인트 정의
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        try:
            image = Image.open(file.file)
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="지원되지 않는 이미지 포맷입니다.")
        
        image_array = preprocess_image(image)

        if model is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
        
        prediction = model.predict(image_array)
        result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
        
        return {"result": result, "score": float(prediction[0][0])}
    
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="예측 중 오류가 발생했습니다.")


# 세션 생성
app.state.session_idx = None

@app.get("/create-session")
def create_session(response: Response):
    session_id = str(uuid.uuid4())
    response.set_cookie(
        key="session_id", 
        value=session_id, 
        httponly=True, 
        secure=False, 
        samesite="Lax")
    return {"session_id": session_id}

@app.get("/get-session")
async def get_session(request: Request, db: Session = Depends(get_db)):
    # 쿠키에서 session_id 가져오기
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID not found in cookies")
    session_info = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_idx": session_info.session_idx}

# 디바이스 정보

# GET 요청에 대한 처리 추가
@app.get("/device-info/")
async def get_device_info():
    return {"message": "This endpoint only accepts POST requests for sending data."}

# 앱 상태를 통해 전역 변수 초기화
@app.post("/device-info/")
async def receive_device_info(
    device_info: DeviceInfo, 
    db: Session = Depends(get_db), 
    session_id: str = Cookie(None)):
    print("Received Device Info:", device_info.dict())
    print("Session ID:", session_id)
    # 세션 ID가 없으면 오류 반환
    if not session_id:
        return {"message": "Session ID is required."}

    # 세션 ID로 기존 데이터 검색
    existing_device_info = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
    if existing_device_info:
    # 기존 데이터가 있을 경우 해당 세션 정보를 반환
        return {"message": "Device information already exists", 
                "data": existing_device_info.session_idx}
    
    # 새로운 디바이스 정보를 데이터베이스에 저장
    try:
        new_device_info = SessionInfo(
            log_device=device_info.userAgent,
            session_id=session_id,
            session_active_duration=None,
            session_expire_dt=None
        )

        db.add(new_device_info)
        db.commit()
        db.refresh(new_device_info)

        app.state.session_idx = new_device_info.session_idx

        return {"message": "Device information saved", "data": new_device_info.session_idx}

    except Exception as e:
        print("Error saving to DB:", e)
        return {"message": "Error saving device information", "error": str(e)}

# 요청 모델 정의
class SessionEndRequest(BaseModel):
    session_expire_dt: str  # 종료 시간 (ISO 포맷으로 받기)

@app.post("/session/end")
async def end_session(request: SessionEndRequest, db: Session = Depends(get_db)):
    print("Request data:", request)  # 요청 데이터 출력

 # 전역 상태에서 session_idx 가져오기
    session_idx = app.state.session_idx
    if not session_idx:
        raise HTTPException(status_code=400, detail="No session ID found. Please create a session first.")

    print("session_idx:", session_idx)

    # 종료 시간 가져오기
    try:
        session_expire_dt = datetime.fromisoformat(request.session_expire_dt)
    except ValueError as e:
        print("Invalid date format error:", e)  # 오류 메시지 출력
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format.")

    # 데이터베이스에서 세션 레코드 조회
    session_record = db.query(SessionInfo).filter(SessionInfo.session_idx == session_idx).first()
    if not session_record:
        raise HTTPException(status_code=404, detail="Session not found")

    # 세션 활성화 기간 계산
    session_created_at = session_record.session_created_at
    active_duration = (session_expire_dt - session_created_at).total_seconds()  # 초 단위로 계산

    # 종료 시간 및 활성화 기간 업데이트
    session_record.session_expire_dt = session_expire_dt
    session_record.session_active_duration = active_duration  # 활성화 기간 저장

    # 데이터베이스에 변경 사항 커밋
    try:
        db.commit()
    except Exception as e:
        db.rollback()  # 오류 발생 시 롤백
        raise HTTPException(status_code=500, detail="Database error occurred.")

    return {
        "message": f"Session {session_idx} has ended successfully",
        "session_end_time": session_expire_dt.isoformat(),
        "active_duration_seconds": active_duration
    }

""" @app.get("/session")
async def get_session(request: Request, db: Session = Depends(get_db)):

    # 전역 상태에서 session_idx 가져오기
    session_idx = app.state.session_idx
    return {"session_idx": session_idx}
 """

app.state.session_idx = None

@app.post("/upload")
async def upload_file(
    image_file: UploadFile = File(...),
    assent_yn: str = Form(...),
    model_pred: float = Form(...),
    db: Session = Depends(get_db)
    
):
    try:
        # 전역 상태에서 session_idx 가져오기
        session_idx = app.state.session_idx

        # 1️⃣ 파일 처리 (Base64로 인코딩)
        content = await image_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")
        
        image = Image.open(io.BytesIO(content))  # 바이너리 데이터를 이미지로 변환
        image_array = preprocess_image(image)  # 이미지를 전처리하여 224x224로 변환
        encoded_image = base64.b64encode(content).decode("utf-8")  # Base64 인코딩
        print(f"🖼️ 예측에 사용된 입력 데이터 크기: {image_array.shape}")
        print("📷 Encoded image size:", len(encoded_image))

        # 2️⃣ 모델을 통해 예측
        if model is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
        
        try:
            prediction = model.predict(image_array)  # 이미지를 예측
            print(f"✅ 예측 결과: {prediction}")
            model_pred = float(prediction[0][0])  # 예측 확률을 추출 (이진 분류의 경우)
        except Exception as e:
            print(f"❌ TensorFlow 예측 오류: {e}")
            raise HTTPException(status_code=500, detail="예측 중 오류가 발생했습니다.")

        # # 3️⃣ 데이터베이스에 엔트리 추가
        # db_entry = UploadInfo(
        #     image_data=encoded_image,
        #     deepfake_data="placeholder_data",  # 예측 결과와 관련된 데이터를 저장
        #     model_pred=model_pred,  # 모델의 예측 결과 저장
        #     session_idx=session_idx,  # 세션 인덱스 저장
        #     assent_yn=assent_yn,  # 동의 여부
        # )

        # 동의 여부에 따라 데이터 처리
        if assent_yn == 'Y':
        # image_backup_info에 데이터 삽입
            backup_entry = ImageBackupInfo(
                deepfake_image_file=encoded_image,
                deepfake_data="placeholder_data",  # 예측 관련 데이터
                session_idx=session_idx,
                model_pred=model_pred,
                assent_yn=assent_yn
            )
            db.add(backup_entry)
        elif assent_yn == 'N':
            # upload_info에 데이터 삽입
            upload_entry = UploadInfo(
                image_data=encoded_image,
                deepfake_data="placeholder_data",
                model_pred=model_pred,
                session_idx=session_idx,
                assent_yn=assent_yn
            )
            db.add(upload_entry)
        else:
            raise HTTPException(status_code=400, detail="assent_yn 값이 유효하지 않습니다. (Y 또는 N)")

        # 변경 사항 저장
        db.commit()

        # 4️⃣ 클라이언트에 예측 결과 및 파일 ID 반환
        result = {"status": "success", "data": prediction.tolist()}
        return JSONResponse(content=result)

    except HTTPException as http_ex:
        raise http_ex  # HTTPException은 다시 발생시킴

    except Exception as e:
        db.rollback()

        print(f"❌ 서버 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
