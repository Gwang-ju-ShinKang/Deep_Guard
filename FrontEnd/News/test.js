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
