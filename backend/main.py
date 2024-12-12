from fastapi import FastAPI, Depends, HTTPException, Request, Response, Cookie
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, UserInfo, UploadInfo, SessionInfo
from itsdangerous import URLSafeTimedSerializer
from pydantic import BaseModel
from typing import List
import uuid
from datetime import datetime, timezone, timedelta
import os
import json
from fastapi import FastAPI, Depends, UploadFile, File, Form

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
@app.get("/items")
def read_items(db: Session = Depends(get_db)):
    items = db.query(UserInfo).all()
    return items

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


 