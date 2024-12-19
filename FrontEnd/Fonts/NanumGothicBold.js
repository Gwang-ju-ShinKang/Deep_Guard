const fs = require("fs");
const path = require("path");

const boldFontPath = path.join("C:\\Users\\smhrd\\Desktop\\WebStudy\\최종\\NanumGothicBold.ttf");
const base64Font = fs.readFileSync(boldFontPath).toString("base64");

fs.writeFileSync("NanumGothicBase64Bold.js", `const base64BoldFont = \`${base64Font}\`;`);
console.log("Base64 데이터가 NanumGothicBase64Bold.js 파일에 저장되었습니다.");
