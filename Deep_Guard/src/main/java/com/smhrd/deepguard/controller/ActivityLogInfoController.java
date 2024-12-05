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

import com.smhrd.deepguard.entity.activity_log_info;

@RestController
@RequestMapping("/activity-logs")
public class ActivityLogInfoController {

    private final ActivityLogInfoController service;

    public ActivityLogInfoController(ActivityLogInfoController service) {
        this.service = service;
    }

    // 모든 로그 조회
    @GetMapping
    public List<activity_log_info> getAllLogs() {
        return service.getAllLogs();
    }

    // log_idx로 특정 로그 조회
    @GetMapping("/{id}")
    public activity_log_info getLogById(@PathVariable("id") long log_idx) {
        return service.getLogById(log_idx);
    }

    // 새로운 로그 추가
    @PostMapping
    public void addLog(@RequestBody activity_log_info logInfo) {
        service.addLog(logInfo);
    }

    // 기존 로그 업데이트
    @PutMapping
    public void updateLog(@RequestBody activity_log_info logInfo) {
        service.updateLog(logInfo);
    }

    // 특정 로그 삭제
    @DeleteMapping("/{id}")
    public void deleteLog(@PathVariable("id") long log_idx) {
        service.deleteLog(log_idx);
    }
}
