from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+pymysql://cgi_24k_data_p3_3:smhrd3@project-db-cgi.smhrd.com:3307/cgi_24k_data_p3_3"

# SQLAlchemy 엔진 생성
engine = create_engine(DATABASE_URL)

# 세션 로컬 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
