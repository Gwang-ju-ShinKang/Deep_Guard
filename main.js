// NanumGothic-Regular 폰트를 Base64 형식으로 추가
const nanumGothicFont = `
<Base64 Encoded Font Content Here>
`;

const fileInput = document.getElementById('file-upload');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const uploadSection = document.getElementById('upload-section');
const loadingSection = document.getElementById('loading-section');
const resultSection = document.getElementById('result-section');
const chart = document.getElementById('chart');
const percentageText = document.getElementById('percentage');
const retryBtn = document.createElement('button');
const pdfButton = document.createElement('button');

// "다시 분석하기" 버튼 추가
retryBtn.textContent = "다시 분석하기";
retryBtn.classList.add('upload-btn');
retryBtn.style.marginTop = "20px";
retryBtn.style.marginRight = "20px";
retryBtn.style.display = "none";
resultSection.appendChild(retryBtn);

// 이미지 업로드 핸들러 (파일 선택)
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
            analyzeBtn.style.display = "inline-block"; // Analyze 버튼 표시
        };
        reader.readAsDataURL(file);
    } else {
        imagePreview.innerHTML = "<p>No image uploaded yet.</p>";
        analyzeBtn.style.display = "none";
    }
});

// 클립보드 이미지 붙여넣기 처리
imagePreview.addEventListener("paste", (event) => {
    const items = event.clipboardData.items;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) {
                handleFileUpload(file);
                analyzeBtn.style.display = "inline-block"; // Analyze 버튼 표시
            }
        }
    }
});

// 파일 처리 함수
function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
        generatePdfButton.style.display = "inline-block"; // PDF 저장 버튼 표시
    };
    reader.readAsDataURL(file);
}

// 분석 버튼 핸들러
analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return alert('이미지를 업로드 해주세요.');

    const formData = new FormData();
    formData.append('file', file);

    uploadSection.classList.remove('active');
    loadingSection.classList.add('active');

    try {
        const response = await fetch('http://127.0.0.1:8000/upload/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`서버 오류 발생: ${response.status}`);
        }

        let result;
        try {
            result = await response.json();
        } catch (e) {
            throw new Error('JSON 파싱에 실패했습니다.');
        }

        console.log('서버 응답 데이터:', result);

        const fakeProbability = result.data && result.data[0] !== undefined
            ? Math.round(result.data[0] * 100)
            : 0;

        animateChart(fakeProbability);

    } catch (error) {
        console.error('에러 발생:', error.message);
        imagePreview.innerHTML = `<p>분석에 실패했습니다. 다시 시도해주세요.</p>`;
        analyzeBtn.style.display = 'inline-block';
    } finally {
        loadingSection.classList.remove('active');
        resultSection.classList.add('active');
    }
});

// 차트 애니메이션 함수
function animateChart(targetPercentage) {
    let currentPercentage = 0;
    const interval = setInterval(() => {
        const angle = currentPercentage * 3.6;

        // 색상 변경
        let color = currentPercentage < 30 ? '#ff5722' :
            currentPercentage < 70 ? '#ffc107' : '#4caf50';

        chart.style.background = `conic-gradient(${color} 0deg ${angle}deg, #ccc ${angle}deg 360deg)`;
        percentageText.textContent = `${currentPercentage}%`;

        if (currentPercentage < targetPercentage) {
            currentPercentage++;
        } else {
            clearInterval(interval);
            retryBtn.textContent = "다시 분석하기";
            retryBtn.style.display = "inline-block";
            resultSection.appendChild(retryBtn);

            pdfButton.textContent = "PDF 저장";
            pdfButton.style.display = "inline-block";
            resultSection.appendChild(pdfButton);
        }
    }, 30);
}

// "다시 분석하기" 버튼 핸들러
retryBtn.addEventListener("click", () => {
    resultSection.classList.remove("active");
    uploadSection.classList.add("active");
    analyzeBtn.style.display = "none";
    imagePreview.innerHTML = "이미지를 올려주세요";
    retryBtn.style.display = "none";
    document.getElementById("generate-pdf").style.display = "none"; // PDF 버튼 숨기기
    chart.style.background = "conic-gradient(#ccc 0deg 360deg)";
    percentageText.textContent = "0";
});

/* 차트 */
const pieCtx = document.getElementById('pieChart').getContext('2d');
new Chart(pieCtx, {
    type: 'pie',
    data: {
        labels: ['딥페이크 포르노', '기타 콘텐츠 (정치, 허위 정보 등)'],
        datasets: [{
            data: [85, 15], // 대한민국 데이터 비율
            backgroundColor: ['#ff5722', '#ccc'],
        }]
    },
    options: {
        plugins: {
            legend: {
                display: true,
                position: 'bottom',
                labels: {
                    font: {
                        family: 'Nanum Gothic',
                    }
                }
            },
        },
    }
});

// 딥페이크 콘텐츠 분포를 보여주는 바 차트
const barCtx = document.getElementById('barChart').getContext('2d');
new Chart(barCtx, {
    type: 'bar',
    data: {
        labels: ['포르노', '정치', '허위 정보', '기타'],
        datasets: [{
            label: '딥페이크 콘텐츠 분포',
            data: [85, 5, 5, 5], // 대한민국 콘텐츠 분포 데이터
            backgroundColor: ['#ff5722', '#ffc107', '#4caf50', '#2196f3'],
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: '비율 (%)',
                    font: {
                        family: 'Nanum Gothic',
                        size: 14,
                    }
                }
            }
        },
        plugins: {
            legend: {
                display: false,
            }
        }
    }
});

document.getElementById("generate-pdf").addEventListener("click", () => {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    // 인증서 배경 이미지 경로
    const img = new Image();
    img.src = 'image/ㅁㅁ.png'; // 인증서 배경 이미지 경로를 정확히 입력해주세요.

    img.onload = function () {
        // 이미지의 원본 크기 유지
        const imgWidth = img.width;
        const imgHeight = img.height;

        // PDF 크기 설정 (A4 기준)
        const pageWidth = 210; // mm
        const pageHeight = 297; // mm

        // 이미지 비율 유지하면서 PDF 크기에 맞게 조정
        const scaleFactor = Math.min(pageWidth / imgWidth, pageHeight / imgHeight);
        const scaledWidth = imgWidth * scaleFactor;
        const scaledHeight = imgHeight * scaleFactor;

        const xOffset = (pageWidth - scaledWidth) / 2; // 중앙 정렬
        const yOffset = (pageHeight - scaledHeight) / 2;

        // PDF에 이미지 추가
        doc.addImage(img, "PNG", xOffset, yOffset, scaledWidth, scaledHeight);

        try {
            // Base64 데이터를 jsPDF에 추가
            doc.addFileToVFS("NanumGothic.ttf", base64Font); // Base64 데이터를 추가
            doc.addFont("NanumGothic.ttf", "NanumGothic", "normal"); // 폰트 등록
            doc.addFileToVFS("NanumGothicBold.ttf", base64BoldFont);
            doc.addFont("NanumGothicBold.ttf", "NanumGothicBold", "normal");
            doc.setFont("NanumGothic"); // 폰트 설정

            // 인증서 제목

            doc.setFontSize(36);
            doc.setFont("NanumGothicBold");
            doc.text("인 증 서", 105, 80 + yOffset, { align: "center" }); // 중앙 상단 대문자로 표시

            // 구분선 추가
            doc.setDrawColor(0); // 검정색 선
            doc.setLineWidth(0.5);
            doc.line(30, 85 + yOffset, 180, 85 + yOffset); // 좌우 선 긋기

            // 주요 정보
            doc.setFontSize(18);
            doc.setFont("NanumGothic", "normal");
            doc.text("본 인증서는 아래의 정보를 확인합니다.", 105, 100 + yOffset, { align: "center" });

            doc.setFontSize(16);
            doc.text("제품 유형: 딥페이크 분석", 30, 130 + yOffset); // 왼쪽 정렬로 제품 유형
            doc.text("모델명: DEEPGUARD001", 30, 140 + yOffset); // 왼쪽 정렬로 모델명

            // 차트 확률 값 추가
            doc.text(`정확도: ${fakeProbability}%`, 30, 150 + yOffset);

            // 인증 날짜 및 유효 기간
            const issueDate = new Date().toLocaleDateString("ko-KR");
            const expiryDate = new Date();
            expiryDate.setDate(expiryDate.getDate() + 30);
            doc.text(`인증 일자: ${issueDate}`, 30, 160 + yOffset); // 왼쪽 정렬로 인증 일자
            doc.text(`유효기간: ${expiryDate.toLocaleDateString("ko-KR")}`, 30, 170 + yOffset);

            // 중앙 강조 문구
            doc.setFontSize(20);
            doc.setFont("NanumGothicBold");
            doc.text("DEEPGUARD의 모델  : DEEPGUARD001(가)", 105, 195 + yOffset, { align: "center" });
            doc.text("이미지를 분석했습니다.", 105, 210 + yOffset, { align: "center" });

            // 하단 안내 문구
            doc.setFontSize(12);
            doc.setFont("NanumGothic", "normal");
            doc.text(
                "이 인증서는 DEEPGUARD 분석 시스템을 통해 자동으로 생성되었습니다.",
                105,
                280 + yOffset,
                { align: "center" }
            );
            doc.setFont("NanumGothicBold", "normal"); // 볼드체 설정
            doc.setFontSize(16);
            doc.text(`발급 일자: ${issueDate}`, 105, 250 + yOffset, { align: "center" }); // 가운데 정렬


            // PDF 저장
            doc.save("딥페이크_분석_결과.pdf");
        } catch (error) {
            console.error("PDF 생성 중 오류 발생:", error.message);
        }
    };

    img.onerror = function () {
        console.error("배경 이미지를 로드할 수 없습니다. 경로를 확인해주세요.");
    };
});

function goToScroll(name) {
    var location = document.querySelector("#" + name).offsetTop;
    window.scrollTo({ top: location - 50 });
}

// 범죄 예방 수칙 온클릭 이벤트
document.querySelectorAll('.rule h2').forEach((title) => {
    title.addEventListener('click', () => {
        const content = title.nextElementSibling;
        if (content.style.display === "block") {
            content.style.display = "none";
        } else {
            content.style.display = "block";
        }
    });
});

 // 페이지 로드 시 세션 확인
 async function checkSession() {
    try {
        // 세션 확인 요청
        const response = await fetch("http://127.0.0.1:8000/get-session", {
            method: "GET",
            credentials: "include", // 쿠키 전송을 허용
        });

        // 세션이 없는 경우
        if (response.status === 400) {
            console.log("세션 없음, 세션 생성 중...");
            const createResponse = await fetch("http://127.0.0.1:8000/create-session", {
                method: "GET",
                credentials: "include", // 쿠키 전송 허용
            });

            if (!createResponse.ok) {
                throw new Error("세션 생성 실패");
            }

            const createData = await createResponse.json();
            console.log("세션 생성 완료:", createData.session_id);
        } else if (response.ok) {
            // 세션이 있는 경우
            const data = await response.json();
            console.log("세션 확인 완료:", data);

        } else {
            throw new Error(`알 수 없는 상태 코드: ${response.status}`);
        }
    } catch (error) {
        console.error("세션 처리 중 오류:", error.message);
    }
}

async function sendDeviceInfoToServer() {
    const deviceInfo = {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/device-info/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(deviceInfo),
            credentials: 'include'
        });

        if (response.ok) {
            const result = await response.json();
            console.log("Server Response:", result);
        } else {
            console.error("Failed to send device info:", response.status);
        }
    } catch (error) {
        console.error("Error sending device info:", error);
    }
}

// 페이지 로드 시 실행
window.onload = () => {
    checkSession();
    sendDeviceInfoToServer();
};