document.addEventListener("DOMContentLoaded", () => {
    const listSection = document.getElementById("listSection");
    const writeSection = document.getElementById("writeSection");
    const pagination = document.querySelector(".pagination");

    // 글쓰기 화면 생성
    function createWriteForm() {
        writeSection.innerHTML = `
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
    }

    // 글쓰기 버튼 클릭 이벤트
    document.querySelector(".write-button").addEventListener("click", () => {
        listSection.style.display = "none";
        pagination.style.display = "none";
        createWriteForm();
        writeSection.style.display = "block";

        document.getElementById("cancelWrite").addEventListener("click", () => {
            writeSection.style.display = "none";
            listSection.style.display = "block";
            pagination.style.display = "flex";
        });

        document.getElementById("writeForm").addEventListener("submit", (e) => {
            e.preventDefault();
            const title = document.getElementById("title").value;
            const author = document.getElementById("author").value;

            const newRow = document.createElement("tr");
            newRow.innerHTML = `
                <td>새글</td>
                <td><a href="#">${title}</a></td>
                <td>${author}</td>
                <td>${new Date().toISOString().split("T")[0]}</td>
                <td>0</td>
            `;
            document.querySelector(".post-list tbody").appendChild(newRow);

            writeSection.style.display = "none";
            listSection.style.display = "block";
            pagination.style.display = "flex";
        });
    });
});
