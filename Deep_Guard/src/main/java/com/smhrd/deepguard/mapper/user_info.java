package com.smhrd.deepguard.mapper;

import java.util.List;

public interface user_info {

    // 모든 사용자 조회
    List<user_info> findAll();

    // user_id로 특정 사용자 조회
    user_info findById(String user_id);

    // 새로운 사용자 삽입
    void insert(user_info userInfo);

    // 기존 사용자 정보 업데이트
    void update(user_info userInfo);

    // user_id로 특정 사용자 삭제
    void delete(String user_id);
}
