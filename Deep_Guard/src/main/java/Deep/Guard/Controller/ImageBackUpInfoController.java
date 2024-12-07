package Deep.Guard.Controller;

import Deep.Guard.Entity.ImageBackUpInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/image-backups")
public class ImageBackUpInfoController {

	@Autowired
	private ImageBackUpInfoController service;

	@GetMapping
	public List<ImageBackUpInfo> getAllBackups() {
		return service.getAllBackups();
	}

	@GetMapping("/{id}")
	public ImageBackUpInfo getBackupById(@PathVariable("id") Long backup_idx) {
		return service.getBackupById(backup_idx);
	}

	@PostMapping
	public void addBackup(@RequestBody ImageBackUpInfo backupInfo) {
		service.addBackup(backupInfo);
	}

	@PutMapping
	public void updateBackup(@RequestBody ImageBackUpInfo backupInfo) {
		service.updateBackup(backupInfo);
	}

	@DeleteMapping("/{id}")
	public void deleteBackup(@PathVariable("id") Long backup_idx) {
		service.deleteBackup(backup_idx);
	}
}
