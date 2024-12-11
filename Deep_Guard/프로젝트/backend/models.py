from sqlalchemy import Column, Integer, String, DateTime, BigInteger, Text, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# 사용자 정보 테이블
class UserInfo(Base):
    __tablename__ = "user_info"

    user_id = Column(String(50), primary_key=True, index=True)
    user_pw = Column(String(128))
    user_contact = Column(String(20))
    user_type = Column(String(10))
    joined_at = Column(DateTime)

# 활동 로그 테이블
class ActivityLogInfo(Base):
    __tablename__ = "activity_log_info"

    log_idx = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(String(50), default="anonymous")
    log_device = Column(String(50))
    log_session = Column(String(300))
    log_time = Column(DateTime)
    report_btn = Column(String(10))
    session_expire_dt = Column(DateTime)


# 파일 업로드 테이블
class UploadInfo(Base):
    __tablename__ = "upload_info"

    image_idx = Column(BigInteger, primary_key=True, index=True)
    image_file = Column(String(1000))
    image_data = Column(Text)
    deepfake_data = Column(Text)
    learning_content = Column(Text)
    model_pred = Column(Numeric)
    created_at = Column(DateTime)
    user_id = Column(String(50), default="anonymous")
    assent_yn = Column(Numeric)

# 이미지 업로드 백업 테이블 
class ImageBackupInfo(Base):
    __tablename__ = "Image_backup_info"

    backup_idx = Column(BigInteger, primary_key=True, index=True)
    original_image_file = Column(String(1000))
    image_data = Column(Text)
    deepfake_data = Column(Text)
    log_device = Column(String(50))
    log_session = Column(String(300))
    created_at = Column(DateTime)
    user_id = Column(String(50), default="anonymous")
    model_pred = Column(Numeric)






