package Deep.Guard.Controller;

import Deep.Guard.Entity.ActivityLogInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/activity-logs")
public class ActivityLogInfoController {

	@Autowired
	private ActivityLogInfoController service;

	@GetMapping
	public List<ActivityLogInfo> getAllLogs() {
		return service.getAllLogs();
	}

	@GetMapping("/{id}")
	public ActivityLogInfo getLogById(@PathVariable("id") Long log_idx) {
		return service.getLogById(log_idx);
	}

	@PostMapping
	public void addLog(@RequestBody ActivityLogInfo logInfo) {
		service.addLog(logInfo);
	}

	@PutMapping
	public void updateLog(@RequestBody ActivityLogInfo logInfo) {
		service.updateLog(logInfo);
	}

	@DeleteMapping("/{id}")
	public void deleteLog(@PathVariable("id") Long log_idx) {
		service.deleteLog(log_idx);
	}
}
