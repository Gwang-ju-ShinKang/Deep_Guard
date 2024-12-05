package com.smhrd.deepguard.mapper;

import java.util.List;

public interface upload_info {

    // 모든 업로드 데이터 조회
    List<upload_info> findAll();

    // image_idx로 특정 업로드 조회
    upload_info findById(long image_idx);

    // 새로운 업로드 데이터 삽입
    void insert(upload_info uploadInfo);

    // 기존 업로드 데이터 업데이트
    void update(upload_info uploadInfo);

    // image_idx로 특정 업로드 데이터 삭제
    void delete(long image_idx);
}
