package Deep.Guard.Mapper;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ImageBackUpInfo {
	List<ImageBackUpInfo> findAll();

	ImageBackUpInfo findById(Long backup_idx);

	void insert(ImageBackUpInfo backupInfo);

	void update(ImageBackUpInfo backupInfo);

	void delete(Long backup_idx);
}
