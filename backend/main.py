from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, UserInfo
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os
import logging

logging.basicConfig(level=logging.DEBUG)
# 데이터베이스 초기화
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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
    
# HTML 페이지 제공
@app.get("/", response_class=HTMLResponse)
async def main():
    with open("h.html") as f:
        return f.read()

# 모델 로드
model_path = r'C:\Users\smhrd\Desktop\git\Deep_Guard\my_model.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_image(file):
    try:
        image = Image.open(io.BytesIO(file.read()))  # 스트림을 BytesIO로 변환
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0  # 이미지를 0~1로 정규화
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise ValueError(f"Could not process image: {str(e)}")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 이미지 타입 검사
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            return {"error": "이미지 파일만 업로드할 수 있습니다."}
        
        # 파일 읽기 (await 추가)
        file_contents = await file.read()  
        image = Image.open(io.BytesIO(file_contents))  # 스트림을 BytesIO로 변환
        
        # 성공 메시지를 반환 (JSON)
        return {
            "filename": file.filename, 
            "message": "이미지 업로드 성공!"
        }
    except Exception as e:
        # 에러 메시지도 명확히 JSON으로 반환
        return {
            "error": f"❌ 서버 오류 발생: {str(e)}"
        }