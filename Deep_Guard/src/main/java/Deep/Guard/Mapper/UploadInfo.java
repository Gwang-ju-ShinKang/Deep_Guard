package Deep.Guard.Mapper;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UploadInfo {
	List<UploadInfo> findAll();

	UploadInfo findById(Long image_idx);

	void insert(UploadInfo uploadInfo);

	void update(UploadInfo uploadInfo);

	void delete(Long image_idx);
}
