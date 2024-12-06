package Deep.Guard.Mapper;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ActivityLogInfo {
	List<ActivityLogInfo> findAll();

	ActivityLogInfo findById(Long log_idx);

	void insert(ActivityLogInfo logInfo);

	void update(ActivityLogInfo logInfo);

	void delete(Long log_idx);
}
