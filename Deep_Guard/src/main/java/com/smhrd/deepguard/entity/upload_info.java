package com.smhrd.deepguard.entity;

import java.math.BigDecimal;
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
public class upload_info {

	private long image_idx;
	private String image_file;
	private String image_data;
	private String deepfake_data;
	private String learning_content;
	private BigDecimal model_pred;
	private LocalDateTime created_at;
	private String user_id;
	private String assent_yn;

}
