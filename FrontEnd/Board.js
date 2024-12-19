document.addEventListener("DOMContentLoaded", () => {
    // DOM 요소 선택
    const listSection = document.getElementById("list"); // 게시글 목록 섹션
    const writeSection = document.getElementById("write"); // 글쓰기 섹션
    const writeButton = document.querySelector(".write-button"); // 글쓰기 버튼
    const pagination = document.querySelector(".pagination"); // 페이지 네이션

    // 글쓰기 화면 생성
    function createWriteForm() {
        const writeFormHTML = `
            <div class="board-header">
                <h1>글쓰기</h1>
            </div>
            <form id="writeForm">
                <div class="form-group">
                    <label for="title">제목</label>
                    <input type="text" id="title" name="title" required>
                </div>
                <div class="form-group">
                    <label for="author">작성자</label>
                    <input type="text" id="author" name="author" required>
                </div>
                <div class="form-group">
                    <label for="content">내용</label>
                    <textarea id="content" name="content" rows="10" required></textarea>
                </div>
                <div class="form-group">
                    <button type="submit" class="submit-btn">저장</button>
                    <button type="button" id="cancelWrite" class="cancel-btn">취소</button>
                </div>
            </form>
        `;
        writeSection.innerHTML = writeFormHTML;
    }

    // 글쓰기 버튼 클릭 시 동작
    writeButton.addEventListener("click", (e) => {
        e.preventDefault();
        listSection.style.display = "none"; // 게시글 목록 숨기기
        pagination.style.display = "none"; // 페이지 네이션 숨기기
        createWriteForm(); // 글쓰기 폼 생성
        writeSection.style.display = "block"; // 글쓰기 섹션 표시

        // 글쓰기 취소 버튼 동작
        const cancelWriteBtn = document.getElementById("cancelWrite");
        cancelWriteBtn.addEventListener("click", (e) => {
            e.preventDefault();
            writeSection.style.display = "none"; // 글쓰기 섹션 숨기기
            listSection.style.display = "block"; // 게시글 목록 표시
            pagination.style.display = "flex"; // 페이지 네이션 표시
        });

        // 글 저장 버튼 동작
        const writeForm = document.getElementById("writeForm");
        writeForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const title = document.getElementById("title").value;
            const author = document.getElementById("author").value;
            const content = document.getElementById("content").value;

            // 게시글 목록에 추가
            const newRow = document.createElement("tr");
            newRow.innerHTML = `
                <td>새글</td>
                <td><a href="#">${title}</a></td>
                <td>${author}</td>
                <td>${new Date().toISOString().split('T')[0]}</td>
                <td>0</td>
            `;
            document.querySelector(".post-list tbody").appendChild(newRow);

            // 초기화 후 목록으로 돌아가기
            writeForm.reset();
            writeSection.style.display = "none";
            listSection.style.display = "block";
            pagination.style.display = "flex";
            alert("글이 저장되었습니다!");
        });
    });
});
