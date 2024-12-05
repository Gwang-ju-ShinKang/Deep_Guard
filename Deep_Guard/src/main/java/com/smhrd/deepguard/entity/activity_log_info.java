package com.smhrd.deepguard.entity;

import java.time.LocalDateTime;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@AllArgsConstructor
@Getter
@Setter
@NoArgsConstructor
@ToString // 객체 안에 있는 필드값들을 문자열로 보여줌
public class activity_log_info {

	private long log_idx;
	private String usdr_id;
	private String log_device;
	private String log_session;
	private LocalDateTime log_time;
	private String report_btn;
	private LocalDateTime session_expire_dt;
	public void insert(activity_log_info testLog) {
		// TODO Auto-generated method stub
		
	}

}
