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
const consentSection = document.getElementById("consent-section");

// "다시 분석하기" 버튼 추가
retryBtn.textContent = "다시 분석하기";
retryBtn.classList.add('upload-btn');
retryBtn.style.marginTop = "20px";
retryBtn.style.marginRight = "20px";
retryBtn.style.display = "none";
resultSection.appendChild(retryBtn);

// PDF 저장 버튼 추가


// 이미지 업로드 핸들러
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
            analyzeBtn.style.display = "inline-block"; // Analyze 버튼 표시
            consentSection.style.display = "block"; // 동의 체크박스 표시
        };
        reader.readAsDataURL(file);
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
                consentSection.style.display = "block"; // 동의 체크박스 표시
            }
        }
    }
});

// 파일 처리 함수
function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
        analyzeBtn.style.display = "inline-block";; // PDF 저장 버튼 표시
    };
    reader.readAsDataURL(file);
}

// 분석 버튼 핸들러
let fakeProbability = 0; // 정확도 값을 저장할 변수
analyzeBtn.addEventListener('click', () => {
    uploadSection.classList.remove('active');
    loadingSection.classList.add('active');

    setTimeout(() => {
        loadingSection.classList.remove('active');
        resultSection.classList.add('active');

        // 차트 애니메이션 실행
        fakeProbability = Math.floor(Math.random() * 101);
        animateChart(fakeProbability);

    }, 3000); // 3초 동안 로딩 상태 유지
});

// 차트 애니메이션 함수
function animateChart(targetPercentage) {
    let currentPercentage = 0;
    const interval = setInterval(() => {
        const angle = currentPercentage * 3.6;

        // 색상 변경 로직
        let color = "#4caf50"; // Default green
        if (currentPercentage < 30) {
            color = "#ff5722"; // Red for low percentages
        } else if (currentPercentage < 70) {
            color = "#ffc107"; // Yellow for mid percentages
        }

        // 차트 배경 업데이트
        chart.style.background = `conic-gradient(${color} 0deg ${angle}deg, #ccc ${angle}deg 360deg)`;
        percentageText.textContent = currentPercentage; // 텍스트 업데이트

        // 목표치에 도달하면 애니메이션 멈추기
        if (currentPercentage >= targetPercentage) {
            clearInterval(interval);

            // "다시 분석하기"와 "PDF 저장" 버튼 표시
            retryBtn.style.display = "inline-block";
            document.getElementById("generate-pdf").style.display = "inline-block";
        } else {
            currentPercentage++; // 퍼센트 증가
        }
    }, 30); // 30ms 간격으로 업데이트
}

// "다시 분석하기" 버튼 핸들러
retryBtn.addEventListener("click", () => {
    resultSection.classList.remove("active");
    uploadSection.classList.add("active");
    analyzeBtn.style.display = "none";
    imagePreview.innerHTML = "이미지를 올려주세요 또는 Ctrl+ V";
    retryBtn.style.display = "none";
    document.getElementById("generate-pdf").style.display = "none"; // PDF 버튼 숨기기
    chart.style.background = "conic-gradient(#ccc 0deg 360deg)";
    consentSection.style.display = "none"; // 동의 섹션 숨김
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


// upload
document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("file-upload");
    const analyzeButton = document.getElementById("analyze-btn");
    const assentCheckbox = document.getElementById("assent-checkbox");
    const imagePreview = document.getElementById("image-preview");
    const uploadSection = document.getElementById("upload-section");
    const loadingSection = document.getElementById("loading-section");
    const resultSection = document.getElementById("result-section");
    const percentageSpan = document.getElementById("percentage");

    // 동의 체크박스 클릭 시 알림
    assentCheckbox.addEventListener("change", () => {
        if (assentCheckbox.checked) {
            const now = new Date();
            alert(`약관에 동의하셨습니다. 동의 시간: ${now.toLocaleString()}`);
        } else {
            alert("약관 동의를 해제하셨습니다.");
        }
    });

    // 세션 ID 가져오기 함수
    async function getSessionIdx() {
        try {
            const response = await fetch("http://127.0.0.1:8000/session", {
                method: "GET",
                credentials: "include",
            });

            if (response.ok) {
                const sessionData = await response.json();
                return sessionData.session_idx; // 응답 구조에 맞게 수정
            } else {
                console.error("세션 정보 가져오기 실패:", response.status);
                return null;
            }
        } catch (error) {
            console.error("세션 정보 요청 중 오류:", error);
            return null;
        }
    }

    // Analyze 버튼 클릭 이벤트
    analyzeButton.addEventListener("click", async () => {
        if (fileInput.files.length === 0) {
            alert("이미지를 업로드해주세요.");
            return;
        }

        const assent_yn = assentCheckbox.checked ? "Y" : "N";
        if (assent_yn === "N") { // 동의하지 않은 경우 처리
            alert("동의하지 않으면 데이터를 저장할 수 없습니다.");
            return;
        }

        // 세션 ID 가져오기
        const session_idx = await getSessionIdx();
        if (!session_idx) {
            alert("세션 정보를 가져오는 데 실패했습니다. 다시 시도해주세요.");
            return;
        }

        // 로딩 상태 활성화
        loadingSection.classList.add("active");

        // FormData 생성
        const formData = new FormData();
        formData.append("image_file", fileInput.files[0]);
        formData.append("assent_yn", assent_yn);
        formData.append("model_pred", "0.987654321");
        formData.append("session_idx", session_idx); // 세션 ID 추가

        try {
            const response = await fetch("http://127.0.0.1:8000/upload", {
                method: "POST",
                body: formData,
                credentials: "include",
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            console.log("서버 응답:", result);

            // 분석 결과 표시
            loadingSection.classList.remove("active"); // 로딩 종료
            resultSection.classList.add("active");
            percentageSpan.textContent = (result.model_pred * 100).toFixed(2); // 결과 표시
        } catch (error) {
            console.error("업로드 중 오류:", error);
            alert("업로드 실패. 서버와 연결할 수 없습니다.");
        }
    });
});
