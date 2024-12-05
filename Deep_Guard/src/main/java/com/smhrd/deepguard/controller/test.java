package com.smhrd.deepguard.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

import com.smhrd.deepguard.mapper.user_info;

@Controller
public class test {
	
	@Autowired
	public static user_info user;
	
	@RequestMapping("/")
	public String homepage() {
		user.findAll();
		return "home";
	}
}
