package Deep.Guard.Mapper;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserInfo {
	List<UserInfo> findAll();

	UserInfo findById(String user_id);

	void insert(UserInfo userInfo);

	void update(UserInfo userInfo);

	void delete(String user_id);
}
