from sqlalchemy import Column, Integer, String, DateTime, BigInteger, Text, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import DECIMAL, TIMESTAMP
from datetime import datetime


Base = declarative_base()

# session info 테이블
class SessionInfo(Base):
    __tablename__ = "session_info"

    session_idx = Column(Integer, primary_key=True, autoincrement=True)  # session_idx를 기본 키로 설정
    log_device = Column(String, nullable=False)
    session_id = Column(String, nullable=False)
    session_created_at = Column(DateTime, nullable=False)
    session_active_duration = Column(Integer, nullable=True)  # 초 단위
    session_expire_dt = Column(DateTime, nullable=False)

# 파일 업로드 테이블
class UploadInfo(Base):
    __tablename__ = "upload_info"
    
    image_idx = Column(Integer, primary_key=True, autoincrement=True, nullable=False)  # INT, PRIMARY KEY, AUTO_INCREMENT
    image_data = Column(Text, nullable=False)  # MEDIUMTEXT
    deepfake_data = Column(Text, nullable=False)  # MEDIUMTEXT
    model_pred = Column(DECIMAL(13, 10), nullable=False)  # DECIMAL(13,10)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)  # TIMESTAMP
    session_idx = Column(Integer, nullable=False)  # INT
    assent_yn = Column(String(1), nullable=False)  # CHAR(1)

# 백업 정보 테이블 
class ImageBackupInfo(Base):
    __tablename__ = "image_backup_info"

    backup_idx = Column(BigInteger, primary_key=True, index=True)
    deepfake_image_file = Column(String(1000))
    deepfake_data = Column(Text)
    session_idx = Column(String(1000))
    created_at = Column(DateTime)
    model_pred = Column(Numeric)

class ImageBackupInfo(Base):
    __tablename__ = "kang"

    backup_idx = Column(BigInteger, primary_key=True, index=True)
    deepfake_image_file = Column(String(1000))
    deepfake_data = Column(Text)
    session_idx = Column(String(1000))
    created_at = Column(DateTime)
    model_pred = Column(Numeric)


    






