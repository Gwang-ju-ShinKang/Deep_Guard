package Deep.Guard.Controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;

import Deep.Guard.Entity.UserInfo;

@Controller
public class test {
	
	@Autowired
	public Deep.Guard.Mapper.UserInfo userMapper;
	
	@RequestMapping("/test")
	public String test(Model model) {
		
		List<Deep.Guard.Mapper.UserInfo> user = userMapper.findAll();
		model.addAttribute("user", user);
		
		return "test";
	}
	
}
