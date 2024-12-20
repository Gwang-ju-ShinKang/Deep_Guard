from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from datetime import datetime, timezone, timedelta
from models import Base, UploadInfo, SessionInfo, DeviceInfo, ImageBackupInfo
from itsdangerous import URLSafeTimedSerializer
from PIL import Image
from pydantic import BaseModel
import numpy as np
import os
import uuid
import io
import tensorflow as tf
import base64

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

# 모델 로드
# 프로젝트의 루트 경로를 base_dir로 설정 (현재 파일의 상위 디렉터리로 이동)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 현재 파일의 상위 디렉터리로 이동
model_path = os.path.join(base_dir, 'BackEnd\Models', 'CNN_model_V2.h5')  # Model 폴더로 경로 생성

print(f"🔍 모델 경로: {model_path}")  # 디버깅을 위해 경로 확인

try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ 모델이 성공적으로 로드되었습니다! (입력 형태: {model.input_shape})")
except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_path}")
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

""" # 📁 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 상위 디렉터리로 이동
model_json_path = os.path.join(base_dir, 'Front_End', 'Models', 'model.json')  # 모델 구조 경로
model_weights_path = os.path.join(base_dir, 'Front_End', 'Models', 'model_weights.h5')  # 가중치 파일 경로

print(f"🔍 모델 JSON 경로: {model_json_path}")
print(f"🔍 모델 가중치 경로: {model_weights_path}")

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

    # 🔥 모델 다시 생성
    model = model_from_json(json.dumps(model_config))
    model.load_weights(model_weights_path)
    print(f"✅ 모델이 성공적으로 로드되었습니다! (입력 형태: {model.input_shape})")

except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {model_json_path} 또는 {model_weights_path}")
    model = None  # 모델 로드 실패 시 None으로 설정
except Exception as e:
    print(f"❌ 모델 로드 중 알 수 없는 오류가 발생했습니다: {e}")
    model = None  # 모델 로드 실패 시 None으로 설정

# 🖼️ 이미지 전처리 함수
def preprocess_image(image: Image.Image) -> np.ndarray:
     # 이미지를 299x299x3 형태로 변환하는 함수 
    try:
        image = image.resize((299, 299))  # 이미지 크기 변경
        if image.mode in ['L', 'RGBA']:  # 그레이스케일(L) 또는 RGBA를 RGB로 변환
            image = image.convert('RGB')

        image_array = np.array(image)  # 이미지 배열로 변환
        if image_array.ndim == 2:  # 흑백 이미지인 경우 3채널로 확장
            image_array = np.stack([image_array] * 3, axis=-1)

        print(f"🖼️ 이미지 배열 크기: {image_array.shape}")
        image_array = image_array / 255.0  # 정규화
        image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가
        return image_array

    except Exception as e:
        print(f"❌ 이미지 전처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="이미지 전처리 중 오류가 발생했습니다.")

# 🚀 FastAPI 앱 생성 및 엔드포인트 정의
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
     # 이미지를 업로드하면 예측 결과 반환 
    try:
        image = Image.open(file.file)  # 이미지 열기
        image_array = preprocess_image(image)  # 이미지 전처리

        if model is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
        
        prediction = model.predict(image_array)
        result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
        
        return {"result": result, "score": float(prediction[0][0])}
    
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="예측 중 오류가 발생했습니다.")
 """


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

        # 파일 처리 (Base64로 인코딩)
        content = await image_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")
        
        # Base64로 이미지 인코딩
        encoded_image = base64.b64encode(content).decode("utf-8")

        # 모델 예측 결과 처리 (기존 로직 포함)
        if model is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
        
        try:
            prediction = model.predict(preprocess_image(Image.open(io.BytesIO(content))))
            print(f"✅ 예측 결과: {prediction}")
            model_pred = float(prediction[0][0])  # 예측 확률 추출
        except Exception as e:
            print(f"❌ 모델 예측 오류: {e}")
            raise HTTPException(status_code=500, detail="예측 중 오류가 발생했습니다.")

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

        return {"status": "success", "message": "데이터가 성공적으로 저장되었습니다."}

    except Exception as e:
        db.rollback()
        print(f"❌ 서버 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
