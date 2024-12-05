package com.smhrd.deepguard.mapper;

import java.util.List;

public interface image_backup_info {

    // 모든 백업 데이터 조회
    List<image_backup_info> findAll();

    // backup_idx로 특정 백업 조회
    image_backup_info findById(long backup_idx);

    // 새로운 백업 데이터 삽입
    void insert(image_backup_info backupInfo);

    // 기존 백업 데이터 업데이트
    void update(image_backup_info backupInfo);

    // backup_idx로 특정 백업 데이터 삭제
    void delete(long backup_idx);
}
