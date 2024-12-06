package Deep.Guard.Controller;

import Deep.Guard.Entity.UserInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/users")
public class UserInfoController {

	@Autowired
	private UserInfoController service;

	@GetMapping
	public List<UserInfo> getAllUsers() {
		return service.getAllUsers();
	}

	@GetMapping("/{id}")
	public UserInfo getUserById(@PathVariable("id") String user_id) {
		return service.getUserById(user_id);
	}

	@PostMapping
	public void addUser(@RequestBody UserInfo userInfo) {
		service.addUser(userInfo);
	}

	@PutMapping
	public void updateUser(@RequestBody UserInfo userInfo) {
		service.updateUser(userInfo);
	}

	@DeleteMapping("/{id}")
	public void deleteUser(@PathVariable("id") String user_id) {
		service.deleteUser(user_id);
	}
}
