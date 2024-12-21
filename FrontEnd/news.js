function initializeLoopingCarousel(trackId, prevBtnId, nextBtnId, autoSlideInterval = 3000) {
  const track = document.getElementById(trackId);
  const slides = Array.from(track.children);
  const slideWidth = slides[0].getBoundingClientRect().width;

  // Clone slides for seamless looping
  slides.forEach(slide => {
    const clone = slide.cloneNode(true);
    track.appendChild(clone); // Append clone at the end
  });

  slides.forEach(slide => {
    const clone = slide.cloneNode(true);
    track.insertBefore(clone, track.firstChild); // Prepend clone at the beginning
  });

  const allSlides = Array.from(track.children); // All slides including clones
  let currentIndex = slides.length; // Start at the original first slide

  // Set track initial position
  track.style.transform = `translateX(-${currentIndex * slideWidth}px)`;

  // Move to slide
  const moveToSlide = (index) => {
    track.style.transition = "transform 0.5s ease-in-out";
    track.style.transform = `translateX(-${index * slideWidth}px)`;
  };

  // Next slide
  const moveToNext = () => {
    currentIndex++;
    moveToSlide(currentIndex);

    if (currentIndex >= allSlides.length - slides.length) {
      setTimeout(() => {
        track.style.transition = "none";
        currentIndex = slides.length; // Reset to the first original slide
        track.style.transform = `translateX(-${currentIndex * slideWidth}px)`;
      }, 500);
    }
  };

  // Previous slide
  const moveToPrev = () => {
    currentIndex--;
    moveToSlide(currentIndex);

    if (currentIndex < slides.length) {
      setTimeout(() => {
        track.style.transition = "none";
        currentIndex = allSlides.length - slides.length * 2; // Reset to the last original slide
        track.style.transform = `translateX(-${currentIndex * slideWidth}px)`;
      }, 500);
    }
  };

  // Buttons
  const prevButton = document.getElementById(prevBtnId);
  const nextButton = document.getElementById(nextBtnId);

  prevButton.addEventListener("click", moveToPrev);
  nextButton.addEventListener("click", moveToNext);

  // Auto-slide
  setInterval(moveToNext, autoSlideInterval);
}

// Initialize the carousels
initializeLoopingCarousel("news-carousel", "news-prev", "news-next");
initializeLoopingCarousel("case-carousel", "prev-case", "next-case");

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