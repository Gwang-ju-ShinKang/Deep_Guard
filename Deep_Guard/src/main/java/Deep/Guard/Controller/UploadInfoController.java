package Deep.Guard.Controller;

import Deep.Guard.Entity.UploadInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/uploads")
public class UploadInfoController {

	@Autowired
	private UploadInfoController service;

	@GetMapping
	public List<UploadInfo> getAllUploads() {
		return service.getAllUploads();
	}

	@GetMapping("/{id}")
	public UploadInfo getUploadById(@PathVariable("id") Long image_idx) {
		return service.getUploadById(image_idx);
	}

	@PostMapping
	public void addUpload(@RequestBody UploadInfo uploadInfo) {
		service.addUpload(uploadInfo);
	}

	@PutMapping
	public void updateUpload(@RequestBody UploadInfo uploadInfo) {
		service.updateUpload(uploadInfo);
	}

	@DeleteMapping("/{id}")
	public void deleteUpload(@PathVariable("id") Long image_idx) {
		service.deleteUpload(image_idx);
	}
}
