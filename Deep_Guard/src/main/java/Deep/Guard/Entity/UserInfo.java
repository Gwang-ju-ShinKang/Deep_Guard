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
public class UserInfo {

	private String user_id;
	private String user_pw;
	private String user_contact;
	private String user_type;
	private String joined_at;

}
