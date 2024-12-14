from fastapi import FastAPI, Depends, HTTPException, Request, Response, Cookie, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from backend.database import SessionLocal, engine
from backend.models import Base, UploadInfo, SessionInfo, ImageBackupInfo
from itsdangerous import URLSafeTimedSerializer
from pydantic import BaseModel
from typing import List
import uuid
from datetime import datetime, timezone, timedelta
import os
import json
from sqlalchemy import func

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

# Secret Key와 Serializer 설정
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
serializer = URLSafeTimedSerializer(SECRET_KEY)

# API 엔드포인트: 모든 데이터 가져오기
@app.get("/items")
def read_items(db: Session = Depends(get_db)):
    items = db.query(SessionInfo).all()
    return items

@app.get("/image")
def read_images(db: Session = Depends(get_db)):
    items = db.query(UploadInfo).all()
    return items

@app.get("/check-sessions/")
def check_sessions(db: Session = Depends(get_db)):
    sessions = db.query(SessionInfo).all()
    return sessions

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
class DeviceInfo(BaseModel):
    userAgent: str
    platform: str
    language: str

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
    # 세션 ID 확인
    if not session_id:
        return {"message": "Session ID is required."}

    # 세션 데이터를 복원
    session_data = serializer.loads(session_id)

    # 시간 계산
    try:
        session_created_at = datetime.fromisoformat(session_data["created_at"])
        session_expire_dt = datetime.fromisoformat(session_data["session_expire_dt"])
        session_active_duration = int((session_expire_dt - session_created_at).total_seconds())
    except Exception as e:
        print("Error parsing times:", e)
        return {"message": "Invalid session data.", "error": str(e)}

    # 디바이스 정보를 데이터베이스에 저장
    try:
        new_device_info = SessionInfo(
            log_device=device_info.userAgent,
            session_id=session_id,
            session_created_at=session_created_at,
            session_active_duration=session_active_duration,
            session_expire_dt=session_expire_dt
        )
        
        db.add(new_device_info)
        db.commit()
        db.refresh(new_device_info)
        
        return {"message": "Device information saved", "data": new_device_info.session_idx}
    except Exception as e:
        print("Error saving device information:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-session-duration/")
def update_session_duration(
    db: Session = Depends(get_db),
    session_id: str = Cookie(None)
):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try:
        session_record = db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()

        if not session_record:
            raise HTTPException(status_code=404, detail="Session not found.")

        # 시간 계산
        session_created_at = session_record.session_created_at
        session_expire_dt = session_record.session_expire_dt

        if session_created_at and session_expire_dt:
            session_active_duration = int((session_expire_dt - session_created_at).total_seconds())
            session_record.session_active_duration = session_active_duration
            db.commit()
            db.refresh(session_record)

            return {"message": "Session duration updated", "data": session_active_duration}
        else:
            raise HTTPException(status_code=400, detail="Invalid session times.")
    except Exception as e:
        print("Error updating session duration:", e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upload")
async def upload_file(
    session_idx: int = Form(...),
    image_file: UploadFile = File(...),
    assent_yn: str = Form(...),
    model_pred: float = Form(...),
    db: Session = Depends(get_db)
):
    try:
        print(f"Using session_idx={session_idx}, assent_yn={assent_yn}, model_pred={model_pred}")
        
        # Database check for session
        session_info = db.query(SessionInfo).filter(SessionInfo.session_idx == session_idx).first()
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found in database.")

        # File processing (dummy for now)
        content = await image_file.read()
        print("File content length:", len(content))
        
        # Add entry to database
        db_entry = UploadInfo(
            image_data="dummy_base64",  # Replace with actual base64 encoding
            deepfake_data="placeholder_data",
            model_pred=model_pred,
            session_idx=session_idx,
            assent_yn=assent_yn,
            created_at=datetime.utcnow()
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        return {"message": "File uploaded successfully", "file_id": db_entry.image_idx}
    except Exception as e:
        print("Error in upload_file:", e)
        raise HTTPException(status_code=500, detail=str(e))
