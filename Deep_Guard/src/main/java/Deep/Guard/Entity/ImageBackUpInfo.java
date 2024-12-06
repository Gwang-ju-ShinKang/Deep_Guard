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
public class ImageBackUpInfo {

	private Long backup_idx;
	private String original_image_file;
	private String image_data;
	private String deepfake_data;
	private String log_device;
	private String log_session;
	private String created_at;
	private String user_id;
	private Double model_pred;

}
