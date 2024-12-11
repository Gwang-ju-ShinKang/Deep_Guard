from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, UserInfo, UploadInfo
from itsdangerous import URLSafeTimedSerializer
from pydantic import BaseModel
from typing import List
import uuid
from datetime import datetime, timezone
import os

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
        "session_id": str(uuid.uuid4()),          # 고유 세션 ID
        "created_at": datetime.now(timezone.utc)  # 세션 생성 시간 (UTC)
    }
    session_id = serializer.dumps(session_data)  # 세션 데이터를 직렬화

    response.set_cookie(
        key="session_id",
        value=session_id,
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