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
public class ActivityLogInfo {

	private Long log_idx;
	private String user_id;
	private String log_device;
	private String log_session;
	private String log_time;
	private String report_btn;
	private String session_expire_dt;

}
