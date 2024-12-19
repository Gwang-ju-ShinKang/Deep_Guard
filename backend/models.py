from sqlalchemy import Column, Integer, String, DateTime, BigInteger, Text, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import DECIMAL, TIMESTAMP
from datetime import datetime


Base = declarative_base()

# 파일 업로드 테이블
class UploadInfo(Base):
    __tablename__ = "upload_info"
    
    image_idx = Column(Integer, primary_key=True, autoincrement=True, nullable=False)  # INT, PRIMARY KEY, AUTO_INCREMENT
    image_data = Column(Text, nullable=False)  # MEDIUMTEXT
    deepfake_data = Column(Text, nullable=False)  # MEDIUMTEXT
    model_pred = Column(DECIMAL(13, 10), nullable=False)  # DECIMAL(13,10)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)  # TIMESTAMP
    session_idx = Column(Integer, ForeignKey('session_info.session_idx'), nullable=False)  # INT
    assent_yn = Column(String(1), nullable=False)  # CHAR(1)

# 이미지 업로드 백업 테이블 
class ImageBackupInfo(Base):
    __tablename__ = "image_backup_info"

    imgage_idx = Column(BigInteger, primary_key=True, index=True)
    deepfake_image_file = Column(Text)
    deepfake_data = Column(Text ,nullable=False)
    session_idx = Column(Integer ,ForeignKey('session_info.session_idx'), nullable=False)
    created_at = Column(DateTime ,nullable=False)
    model_pred = Column(Numeric ,nullable=False)

# session info 테이블
class SessionInfo(Base):
    __tablename__ = "session_info"

    session_idx = Column(Integer, primary_key=True, autoincrement=True)  # session_idx를 기본 키로 설정
    log_device = Column(Text, nullable=False)
    session_id = Column(Text, nullable=False)
    session_created_at = Column(DateTime, nullable=False)
    session_expire_dt = Column(DateTime, nullable=False)
    session_active_duration = Column(DateTime, nullable=False)  # 초 단위
    






