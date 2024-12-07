from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserInfo(Base):
    __tablename__ = "user_info"

    user_id = Column(String(100), primary_key=True, index=True)
    user_pw = Column(String(100))
    user_contact = Column(String(100))
    user_type = Column(String(100))
    joined_at = Column(DateTime)



