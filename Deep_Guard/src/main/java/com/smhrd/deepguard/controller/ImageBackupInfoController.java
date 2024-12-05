package com.smhrd.deepguard.controller;

import java.util.List;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.smhrd.deepguard.entity.image_backup_info;

@RestController
@RequestMapping("/image-backups")
public class ImageBackupInfoController {

    private final ImageBackupInfoController service;

    public ImageBackupInfoController(ImageBackupInfoController service) {
        this.service = service;
    }

    // 모든 백업 데이터 조회
    @GetMapping
    public List<image_backup_info> getAllBackups() {
        return service.getAllBackups();
    }

    // backup_idx로 특정 백업 데이터 조회
    @GetMapping("/{id}")
    public image_backup_info getBackupById(@PathVariable("id") long backup_idx) {
        return service.getBackupById(backup_idx);
    }

    // 새로운 백업 데이터 추가
    @PostMapping
    public void addBackup(@RequestBody image_backup_info backupInfo) {
        service.addBackup(backupInfo);
    }

    // 기존 백업 데이터 업데이트
    @PutMapping
    public void updateBackup(@RequestBody image_backup_info backupInfo) {
        service.updateBackup(backupInfo);
    }

    // 특정 백업 데이터 삭제
    @DeleteMapping("/{id}")
    public void deleteBackup(@PathVariable("id") long backup_idx) {
        service.deleteBackup(backup_idx);
    }
}
