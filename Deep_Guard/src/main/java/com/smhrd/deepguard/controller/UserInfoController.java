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

import com.smhrd.deepguard.entity.user_info;

@RestController
@RequestMapping("/users")
public class UserInfoController {

    private final UserInfoController service;

    public UserInfoController(UserInfoController service) {
        this.service = service;
    }

    // 모든 사용자 조회
    @GetMapping
    public List<user_info> getAllUsers() {
        return service.getAllUsers();
    }

    // user_id로 특정 사용자 조회
    @GetMapping("/{id}")
    public user_info getUserById(@PathVariable("id") String user_id) {
        return service.getUserById(user_id);
    }

    // 새로운 사용자 추가
    @PostMapping
    public void addUser(@RequestBody user_info userInfo) {
        service.addUser(userInfo);
    }

    // 기존 사용자 업데이트
    @PutMapping
    public void updateUser(@RequestBody user_info userInfo) {
        service.updateUser(userInfo);
    }

    // 특정 사용자 삭제
    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable("id") String user_id) {
        service.deleteUser(user_id);
    }
}
