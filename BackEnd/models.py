from sqlalchemy import Column, Integer, String, DateTime, BigInteger, Text, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import DECIMAL, TIMESTAMP
from datetime import datetime
from sqlalchemy.sql import func
from pydantic import BaseModel

Base = declarative_base()

# 파일 업로드 테이블
class UploadInfo(Base):
    __tablename__ = "upload_info"
    
    image_idx = Column(Integer, primary_key=True, autoincrement=True, nullable=False)  # INT, PRIMARY KEY, AUTO_INCREMENT
    image_data = Column(Text, nullable=False)  # MEDIUMTEXT
    deepfake_data = Column(Text, nullable=False)  # MEDIUMTEXT
    model_pred = Column(DECIMAL(13, 10), nullable=False)  # DECIMAL(13,10)
    session_created_at = Column(DateTime, server_default=func.now(), nullable=False)
    session_idx = Column(Integer, nullable=False)  # INT
    assent_yn = Column(String(1), nullable=False)  # CHAR(1)

# 이미지 업로드 백업 테이블 
class ImageBackupInfo(Base):
    __tablename__ = "image_backup_info"

    # 테이블 컬럼 정의
    backup_idx = Column(Integer, primary_key=True, autoincrement=True, nullable=False)  # Primary Key
    deepfake_image_file = Column(Text, nullable=True)  # MEDIUMTEXT (Base64 인코딩된 이미지 데이터)
    deepfake_data = Column(Text, nullable=True)  # MEDIUMTEXT (딥페이크 관련 데이터)
    session_idx = Column(Integer, nullable=True)  # 세션 식별자
    created_at = Column(DateTime, server_default=func.now(), nullable=True)  # 레코드 생성 시간
    model_pred = Column(Numeric(13, 10), nullable=True)  # DECIMAL(13,10) (모델 예측 확률값)
    assent_yn = Column(String(1), nullable=False)  # CHAR(1)


class SessionInfo(Base):
    __tablename__ = "session_info"

     # Primary Key
    session_idx = Column(Integer, primary_key=True, autoincrement=True)

    # 일반 컬럼
    log_device = Column(Text, nullable=True)  # 로그 디바이스 정보
    session_id = Column(Text, nullable=True)  # 세션 ID
    session_active_duration = Column(Integer, nullable=True)  # 세션 활성 시간 (초 단위)
    session_expire_dt = Column(DateTime, nullable=True)  # 세션 만료 시간
    session_created_at = Column(DateTime, server_default=func.now(), nullable=False)  # 생성 시간 (자동)

    def __repr__(self):
        return f"<SessionInfo(session_idx={self.session_idx}, session_id='{self.session_id}')>"

# Pydantic 모델 정의
class DeviceInfo(BaseModel):
    userAgent: str
    platform: str
    language: str    






