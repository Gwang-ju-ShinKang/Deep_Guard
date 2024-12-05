package com.smhrd.deepguard.mapper;

import java.util.List;

public interface activity_log_info {

    // 모든 로그 조회
    List<activity_log_info> findAll();

    // log_idx로 특정 로그 조회
    activity_log_info findById(long log_idx);

    // 새로운 로그 삽입
    void insert(activity_log_info logInfo);

    // 기존 로그 업데이트
    void update(activity_log_info logInfo);

    // log_idx로 특정 로그 삭제
    void delete(long log_idx);
    
    
}
