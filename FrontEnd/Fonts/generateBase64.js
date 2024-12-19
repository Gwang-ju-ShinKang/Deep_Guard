const fs = require("fs");
const path = require("path");

// 폰트 파일 경로 (반영된 경로)
const fontPath = path.join("C:\\Users\\smhrd\\Desktop\\WebStudy\\NanumGothic.ttf");

// Base64 변환
const base64Font = fs.readFileSync(fontPath).toString("base64");

// Base64 결과를 파일로 저장
fs.writeFileSync("NanumGothicBase64.js", `const base64Font = \`${base64Font}\`;`);
console.log("Base64 데이터가 NanumGothicBase64.js 파일에 저장되었습니다.");