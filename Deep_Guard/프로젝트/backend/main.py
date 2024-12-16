from fastapi import FastAPI, Depends, HTTPException, Request, Response, Cookie
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, UploadInfo, SessionInfo, DeviceInfo
from itsdangerous import URLSafeTimedSerializer
from pydantic import BaseModel
from typing import List
import uuid
from datetime import datetime, timezone, timedelta
import os
import json
from fastapi import FastAPI, Depends, UploadFile, File, Form
import base64


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
    session_id = str(uuid.uuid4())

    response.set_cookie(
        key="session_id",
        value=session_id,  # 세션 ID 값
        httponly=True,
        secure=False,  # HTTPS 환경에서는 True로 설정
        samesite="Lax"
    )

    return {"session_id": session_id}

@app.get("/get-session")
async def get_session(request: Request, db: Session = Depends(get_db)):
    # 쿠키에서 session_id 가져오기
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID not found in cookies")
    
    # 데이터베이스에서 session_id를 사용하여 세션 정보 조회
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
app.state.session_idx = None

@app.post("/device-info/")
async def receive_device_info(
    device_info: DeviceInfo,
    db: Session = Depends(get_db),
    session_id: str = Cookie(None),
):
    print("Received Device Info:", device_info.dict())
    print("Session ID:", session_id)

    # 세션 ID가 없으면 오류 반환
    if not session_id:
        return {"message": "Session ID is required."}

    # 세션 ID로 기존 데이터 검색
    existing_device_info = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()
    
    if existing_device_info:
        # 기존 데이터가 있을 경우 해당 세션 정보를 반환
        return {"message": "Device information already exists", "data": existing_device_info.session_idx}

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


# 세션 종료시 활동시간

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



# upload

@app.get("/session")
async def get_session(request: Request, db: Session = Depends(get_db)):

    # 전역 상태에서 session_idx 가져오기
    session_idx = app.state.session_idx
    
    return {"session_idx": session_idx}


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

        # 파일 처리
        content = await image_file.read()
        encoded_image = base64.b64encode(content).decode("utf-8")  # Base64 인코딩
        print("Encoded image size:", len(encoded_image))

        # 데이터베이스에 엔트리 추가
        db_entry = UploadInfo(
            image_data=encoded_image,
            deepfake_data="placeholder_data",  # 실제 데이터로 교체 필요
            model_pred=model_pred,
            session_idx=session_idx,
            assent_yn=assent_yn,
            created_at=datetime.utcnow()
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        return {"message": "File uploaded successfully", "file_id": db_entry.image_idx}

    except HTTPException as http_ex:
        raise http_ex  # HTTPException은 다시 발생시킴
    except Exception as e:
        print("Error in upload_file:", e)
        raise HTTPException(status_code=500, detail=str(e))