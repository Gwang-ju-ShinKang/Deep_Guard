package Deep.Guard.Entity;

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
public class UploadInfo {

	private Long image_idx;
	private String image_file;
	private String image_data;
	private String deepfake_data;
	private String learning_content;
	private Double model_pred;
	private String created_at;
	private String user_id;
	private String assent_yn;

}
