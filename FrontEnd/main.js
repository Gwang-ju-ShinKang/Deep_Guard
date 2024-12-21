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
const assentCheckbox = document.getElementById("assent-checkbox");

document.querySelectorAll('.dropdown').forEach(dropdown => {
    let isHovered = false; // 마우스가 메뉴에 올라간 상태 추적

    // 메뉴 항목에 마우스를 올리면 드롭다운 열기
    dropdown.addEventListener('mouseenter', function () {
        this.classList.add('open'); // 메뉴에 마우스를 올리면 드롭다운 열기
        isHovered = true; // 마우스가 메뉴 항목 위에 있을 때
    });

    // 드롭다운 메뉴 내부에서 마우스가 벗어날 때만 닫기
    dropdown.querySelector('.dropdown-content').addEventListener('mouseleave', function () {
        this.parentElement.classList.remove('open'); // 드롭다운에서 마우스가 벗어나면 닫기
        isHovered = false; // 마우스가 벗어났음을 추적
    });

    // 드롭다운 메뉴 항목에서 벗어나도 드롭다운이 고정되도록
    dropdown.addEventListener('mouseleave', function () {
        if (!isHovered) {
            this.classList.remove('open'); // 마우스가 메뉴와 드롭다운 둘 다 벗어나면 닫기
        }
    });
});

document.getElementById('file-upload').addEventListener('change', function (event) {
    const file = event.target.files[0];

    if (file) {
        // 첫 번째 "사진 선택" 버튼 숨기기
        document.querySelector('label[for="file-upload"]').style.display = 'none';

        // btnSection 보이게 하기
        document.querySelector('.btnSection').style.display = 'flex';

        // 미리보기 이미지 업데이트
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById('preview-img').src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});

document.getElementById('file-upload-file').addEventListener('change', function (event) {
    const file = event.target.files[0];

    if (file) {
        // 첫 번째 "사진 선택" 버튼 숨기기
        document.querySelector('label[for="file-upload-file"]').style.display = 'none';

        // btnSection 보이게 하기
        document.querySelector('.btnSection-file').style.display = 'flex';

        // 선택된 파일에 대한 미리보기 (파일명 표시)
        const filePreview = document.getElementById('file-preview');

        // 파일명과 이미지 추가
        filePreview.innerHTML = `
            <img class="voicePreview" src="../FrontEnd/Images/voiceImage.png" alt="파일 아이콘">
            파일명: ${file.name} `;
    }
});

// "다시 분석하기" 버튼 추가
retryBtn.textContent = "다시 분석하기";
retryBtn.classList.add('upload-btn');
retryBtn.style.marginTop = "20px";
retryBtn.style.marginRight = "20px";
retryBtn.style.display = "none";
resultSection.appendChild(retryBtn);

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
    // 파일 입력 필드 초기화
    fileInput.value = ""; // 파일 입력 필드의 value를 초기화하여 같은 파일도 업로드 가능하도록 설정

    // 동의 체크박스 초기화
    // const assentCheckbox = document.getElementById("assent-checkbox");
    if (assentCheckbox) {
        assentCheckbox.checked = false; // 체크박스 해제
        consentSection.style.display = "none"; // 동의 섹션 숨기기
    }

    // fakeProbability와 currentPercentage 초기화 (새 분석을 위한 준비)
    fakeProbability = 0; // 이전 결과 초기화
    currentPercentage = 0; // 차트 애니메이션을 위한 currentPercentage 초기화
    chart.style.background = "conic-gradient(#ccc 0deg 360deg)"; // 차트 초기화
    percentageText.textContent = "0"; // 차트 텍스트 초기화

    cancelAnimationFrame(animationFrameId); // 애니메이션 중단
});

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
document.addEventListener("paste", (event) => {
    const items = event.clipboardData.items;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) {
                // 파일을 fileInput에 추가
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files; // fileInput의 파일 목록 업데이트

                handleFileUpload(file); // 파일 업로드 처리
                analyzeBtn.style.display = "inline-block"; // Analyze 버튼 표시
                consentSection.style.display = "block"; // 동의 체크박스 표시
                break; // 첫 번째 이미지 파일만 처리
            }
        }
    }
});

// 파일 처리 함수
function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
        analyzeBtn.style.display = "inline-block"; // Analyze 버튼 표시
    };
    reader.readAsDataURL(file);
}

// 분석 버튼 핸들러
//  let fakeProbability = 0; // 정확도 값을 저장할 변수
analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return alert('이미지를 업로드 해주세요.');

    // 동의 체크박스 값 확인
    const assentCheckbox = document.getElementById("assent-checkbox"); // 동의 체크박스 ID
    const assentYn = assentCheckbox && assentCheckbox.checked ? 'Y' : 'N'; // 체크 여부에 따라 Y 또는 N 설정

    const formData = new FormData();
    formData.append('image_file', file); // 'file' 대신 'image_file'로 변경 (FastAPI에 맞춤)
    formData.append('assent_yn', assentYn); // assent_yn 추가 (동의 여부)
    formData.append('model_pred', '0.0'); // model_pred 추가 (필수) 

    uploadSection.classList.remove('active');
    loadingSection.classList.add('active');

    try {
        const response = await fetch('http://127.0.0.1:8000/upload', {
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
    setTimeout(() => {
        loadingSection.classList.remove('active');
        resultSection.classList.add('active');

    }, 30000); // 3초 동안 로딩 상태 유지
});

function animateChart(targetPercentage) {
    currentPercentage = 0;
    function updateChart() {
        const angle = currentPercentage * 3.6;
        let color = currentPercentage < 30 ? "#ff5722"
            : currentPercentage < 70 ? "#ffc107"
                : "#4caf50";
        chart.style.background = `conic-gradient(${color} 0deg ${angle}deg, #ccc ${angle}deg 360deg)`;
        percentageText.textContent = `${currentPercentage}`;

        if (currentPercentage < targetPercentage) {
            currentPercentage++;
            animationFrameId = requestAnimationFrame(updateChart);
        } else {
            retryBtn.style.display = "inline-block";
            document.getElementById("generate-pdf").style.display = "inline-block";
        }
    }

    animationFrameId = requestAnimationFrame(updateChart);
}

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
    img.src = './Images/ㅁㅁ.png'; // 인증서 배경 이미지 경로를 정확히 입력해주세요.

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
            doc.text(`정확도: ${currentPercentage}%`, 30, 150 + yOffset);

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

// 페이지 로드 시 해시가 있는 경우 해당 섹션으로 스크롤
window.onload = function () {
    const hash = window.location.hash; // URL의 해시 가져오기
    if (hash) {
        const targetElement = document.querySelector(hash);
        if (targetElement) {
            const offsetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - 50; // 헤더 높이를 고려하여 조정
            window.scrollTo({
                top: offsetPosition,
                behavior: "smooth" // 부드러운 스크롤 효과
            });
        }
    }
};

// 모달 열기 시 페이지 스크롤 비활성화
function disableScroll() {
    document.body.style.overflow = "hidden"; // 페이지 스크롤 비활성화
}

// 모달 닫기 시 페이지 스크롤 활성화
function enableScroll() {
    document.body.style.overflow = ""; // 페이지 스크롤 복구
}

// 모달창
document.addEventListener("DOMContentLoaded", () => {
    const assentCheckbox = document.getElementById("assent-checkbox");
    const termsModal = document.getElementById("terms-modal");
    const modalConfirmBtn = document.getElementById("confirm-btn");
    const closeTermsBtn = document.getElementById("close-terms");
    const allAgreeCheckbox = document.getElementById("all-agree-checkbox");
    const radios = document.querySelectorAll("input[type=radio]");


    // 모달 초기화 함수
    function resetModal() {
        allAgreeCheckbox.checked = false; // 전체 동의 체크박스 해제
        radios.forEach(radio => {
            radio.checked = false; // 모든 라디오 버튼 체크 해제
        });
        modalConfirmBtn.disabled = true; // 확인 버튼 비활성화
    }

    // 동의 체크박스 클릭 시 모달 열기
    assentCheckbox.addEventListener("change", () => {
        if (assentCheckbox.checked) {
            resetModal(); // 모달 상태 초기화
            termsModal.style.display = "flex"; // 모달 열기
            disableScroll(); // 스크롤 비활성화
        } else {
            alert("동의를 해제하셨습니다.");
        }
    });

    // 모달에서 "확인" 버튼 클릭 시
    modalConfirmBtn.addEventListener("click", () => {
        termsModal.style.display = "none"; // 모달 닫기
        enableScroll(); // 스크롤 활성화
        alert("약관에 동의하셨습니다.");
    });

    // 모달 닫기 버튼 클릭 시
    closeTermsBtn.addEventListener("click", () => {
        termsModal.style.display = "none"; // 모달 닫기
        assentCheckbox.checked = false; // 동의 체크박스 해제
        enableScroll(); // 스크롤 활성화
        resetModal(); // 모달 초기화
    });

    // 모달 외부 클릭 시 닫기
    termsModal.addEventListener("click", (event) => {
        if (event.target === termsModal) {
            termsModal.style.display = "none";
            assentCheckbox.checked = false; // 동의 체크박스 해제
            resetModal(); // 모달 초기화
        }
    });

    // 전체 동의 체크박스 이벤트
    allAgreeCheckbox.addEventListener("change", () => {
        const isChecked = allAgreeCheckbox.checked;

        document.querySelectorAll(".terms-section").forEach(section => {
            const inputs = section.querySelectorAll("input[type=radio]");
            if (isChecked) {
                inputs[0].checked = true; // "동의함" 선택
            } else {
                inputs.forEach(input => {
                    input.checked = false; // 모든 라디오 버튼 체크 해제
                });
            }
        });

        updateConfirmButtonState();
    });

    // 개별 약관 동의 이벤트
    radios.forEach(radio => {
        radio.addEventListener("change", () => {
            updateConfirmButtonState();
            if (!allTermsAgreed()) {
                allAgreeCheckbox.checked = false; // 개별적으로 동의 해제 시 전체 동의 해제
            }
        });
    });

    // 약관 동의 상태 확인 및 버튼 활성화 업데이트
    function updateConfirmButtonState() {
        const allChecked = allTermsAgreed();
        modalConfirmBtn.disabled = !allChecked;
    }

    // 모든 필수 항목이 동의되었는지 확인
    function allTermsAgreed() {
        return [...document.querySelectorAll(".terms-section")].every(section => {
            const inputs = section.querySelectorAll("input[type=radio]");
            return [...inputs].some(input => input.checked && input.value === "agree");
        });
    }
});

// 페이지 로드 시 세션 확인
async function checkSession() {
    try {
        // 쿠키에서 세션 ID 가져오기
        const sessionIdCookie = document.cookie.split('; ').find(row => row.startsWith('session_id='));

        if (!sessionIdCookie) {
            console.log("세션 없음, 세션 생성 중...");
            const createResponse = await fetch("http://127.0.0.1:8000/create-session", {
                method: "GET",
                credentials: "include",
            });

            if (!createResponse.ok) {
                throw new Error("세션 생성 실패");
            }

            const createData = await createResponse.json();
            console.log("세션 생성 완료:", createData.session_id); // 새로운 세션 ID 출력
        } else {
            // 쿠키에서 세션 ID가 존재하는 경우
            const sessionId = sessionIdCookie.split('=')[1];
            console.log("세션 확인 완료:", sessionId); // 기존 세션 ID 출력

            // 기존 세션 ID로 /get-session 호출하여 세션 정보 가져오기
            const getSessionResponse = await fetch("http://127.0.0.1:8000/get-session", {
                method: "GET",
                credentials: "include", // 쿠키 포함
            });

            if (!getSessionResponse.ok) {
                throw new Error("세션 정보 가져오기 실패");
            }

            const sessionData = await getSessionResponse.json();
            console.log("세션 정보:", sessionData); // 세션 정보 출력
        }
    } catch (error) {
        console.error("세션 처리 중 오류:", error.message);
    }
}

// 장치 정보 전송 함수
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

// 페이지 로드 시 장치 정보 전송
// 페이지 로드 시 세션 확인
document.addEventListener("DOMContentLoaded", checkSession);
document.addEventListener("DOMContentLoaded", sendDeviceInfoToServer);

// 한국 시간으로 ISO 형식의 문자열 반환
function getKSTTimeISO() {
    const now = new Date();
    const offset = 9 * 60; // UTC+9 (분 단위)
    const kstTime = new Date(now.getTime() + offset * 60000);
    return kstTime.toISOString().slice(0, 19); // ISO 문자열에서 밀리초와 Z 제거
}

// 세션 종료 시간 전송
async function sendSessionEndTime() {
    const sessionEndTime = getKSTTimeISO(); // 한국 시간 가져오기
    console.log("한국 시간:", sessionEndTime); // 확인용 로그

    try {
        const response = await fetch("http://127.0.0.1:8000/session/end", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                session_expire_dt: sessionEndTime, // 한국 시간 전송
            }),
            credentials: "include",
        });

        if (response.ok) {
            console.log("세션 종료 정보 전송 성공");
        } else {
            const errorData = await response.json();
            console.error("세션 종료 정보 전송 실패:", response.status, errorData);
        }
    } catch (error) {
        console.error("세션 종료 정보 전송 중 오류:", error.message);
    }
}

// 창 종료 시 세션 종료 정보 전송
window.addEventListener("beforeunload", (event) => {
    sendSessionEndTime().then(() => {
        event.returnValue = ""; // Chrome에서 사용자에게 종료 경고 표시
    });
});