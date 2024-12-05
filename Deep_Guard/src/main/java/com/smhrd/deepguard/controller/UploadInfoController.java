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

import com.smhrd.deepguard.entity.upload_info;

@RestController
@RequestMapping("/uploads")
public class UploadInfoController {

    private final UploadInfoController service;

    public UploadInfoController(UploadInfoController service) {
        this.service = service;
    }

    // 모든 업로드 데이터 조회
    @GetMapping
    public List<upload_info> getAllUploads() {
        return service.getAllUploads();
    }

    // image_idx로 특정 업로드 데이터 조회
    @GetMapping("/{id}")
    public upload_info getUploadById(@PathVariable("id") long image_idx) {
        return service.getUploadById(image_idx);
    }

    // 새로운 업로드 데이터 추가
    @PostMapping
    public void addUpload(@RequestBody upload_info uploadInfo) {
        service.addUpload(uploadInfo);
    }

    // 기존 업로드 데이터 업데이트
    @PutMapping
    public void updateUpload(@RequestBody upload_info uploadInfo) {
        service.updateUpload(uploadInfo);
    }

    // 특정 업로드 데이터 삭제
    @DeleteMapping("/{id}")
    public void deleteUpload(@PathVariable("id") long image_idx) {
        service.deleteUpload(image_idx);
    }
}
