/**
 * StreamGVE Project Page - Simple Scripts
 */

// Copy BibTeX to clipboard
function copyBibtex() {
    const bibtexCode = document.querySelector('.bibtex-box code');
    const copyBtn = document.querySelector('.copy-btn');

    if (!bibtexCode || !copyBtn) return;

    const text = bibtexCode.textContent;

    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.style.background = 'rgba(34, 197, 94, 0.3)';
        copyBtn.style.borderColor = 'rgba(34, 197, 94, 0.5)';

        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.background = '';
            copyBtn.style.borderColor = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        copyBtn.textContent = 'Failed';
        setTimeout(() => {
            copyBtn.textContent = 'Copy';
        }, 2000);
    });
}

// Toggle Technical details expand/collapse
function toggleTechnicalDetails() {
    const header = document.querySelector('.technical-details-header');
    const content = document.querySelector('.technical-details-content');
    if (header && content) {
        header.classList.toggle('collapsed');
        content.classList.toggle('collapsed');
    }
}

// Smooth scroll for anchor links (if browser doesn't support CSS smooth scroll)
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Lazy load videos when they come into view (optional optimization)
if ('IntersectionObserver' in window) {
    const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                if (video.dataset.src) {
                    video.src = video.dataset.src;
                    video.removeAttribute('data-src');
                }
                videoObserver.unobserve(video);
            }
        });
    }, {
        rootMargin: '50px'
    });

    document.querySelectorAll('video[data-src]').forEach(video => {
        videoObserver.observe(video);
    });
}

// ==================== VIDEO COMPARISON SLIDER ====================

class VideoCompareSlider {
    constructor(container) {
        this.container = container;
        this.slider = container.querySelector('.compare-slider');
        this.handle = container.querySelector('.compare-handle');
        this.videoFg = container.querySelector('.video-fg');
        this.videoBg = container.querySelector('.video-bg');
        this.isHovering = false;

        this.init();
    }

    init() {
        // Hover-based slider movement
        this.container.addEventListener('mouseenter', () => {
            this.isHovering = true;
        });

        this.container.addEventListener('mouseleave', () => {
            this.isHovering = false;
        });

        this.container.addEventListener('mousemove', (e) => {
            if (this.isHovering) {
                this.updateSliderPosition(e);
            }
        });

        // Touch events (for mobile)
        this.container.addEventListener('touchstart', (e) => {
            this.updateSliderPosition(e);
        }, { passive: true });

        this.container.addEventListener('touchmove', (e) => {
            this.updateSliderPosition(e);
        }, { passive: true });

        // Synchronize videos
        this.syncVideos();
    }

    updateSliderPosition(e) {
        const rect = this.container.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        let percentage = ((clientX - rect.left) / rect.width) * 100;

        // Clamp between 0 and 100
        percentage = Math.max(0, Math.min(100, percentage));

        // Update slider position
        this.slider.style.left = percentage + '%';

        // Update clip-path on foreground video
        if (this.videoFg) {
            this.videoFg.style.clipPath = `inset(0 ${100 - percentage}% 0 0)`;
        }
    }

    syncVideos() {
        if (!this.videoFg || !this.videoBg) return;

        // Store playing state for visibility change handling
        this.wasPlaying = false;

        // Ensure both videos are loaded before playing
        const playVideos = () => {
            const fgPromise = this.videoFg.play();
            const bgPromise = this.videoBg.play();

            // Handle autoplay restrictions gracefully
            if (fgPromise) fgPromise.catch(() => {});
            if (bgPromise) bgPromise.catch(() => {});
        };

        // Wait for both videos to be ready
        let loadedCount = 0;
        const onCanPlay = () => {
            loadedCount++;
            if (loadedCount >= 2) {
                playVideos();
            }
        };

        this.videoFg.addEventListener('canplay', onCanPlay, { once: true });
        this.videoBg.addEventListener('canplay', onCanPlay, { once: true });

        // Fallback: try to play after a short delay anyway
        setTimeout(playVideos, 500);

        // Sync play/pause
        const syncPlay = (source, target) => {
            source.addEventListener('play', () => target.play());
            source.addEventListener('pause', () => target.pause());
            source.addEventListener('seeking', () => {
                if (Math.abs(source.currentTime - target.currentTime) > 0.1) {
                    target.currentTime = source.currentTime;
                }
            });
        };

        syncPlay(this.videoFg, this.videoBg);
        syncPlay(this.videoBg, this.videoFg);

        // Mark container as having videos
        this.container.classList.add('has-videos');
    }
}

// Initialize all video comparison sliders
document.addEventListener('DOMContentLoaded', () => {
    const allVideos = [];

    document.querySelectorAll('.video-compare-container').forEach(container => {
        const slider = new VideoCompareSlider(container);
        // Store video references for visibility handling
        if (slider.videoFg) allVideos.push(slider.videoFg);
        if (slider.videoBg) allVideos.push(slider.videoBg);
    });

    // Initialize all carousels
    document.querySelectorAll('.video-carousel').forEach(carousel => {
        new VideoCarousel(carousel);
    });

    // Handle page visibility change - resume videos when user returns
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            // Small delay to ensure browser is ready
            setTimeout(() => {
                document.querySelectorAll('video').forEach(video => {
                    if (video.paused) {
                        video.play().catch(() => {});
                    }
                });
            }, 100);
        }
    });
});

// ==================== VIDEO CAROUSEL ====================

class VideoCarousel {
    constructor(container) {
        this.container = container;
        this.track = container.querySelector('.carousel-track');
        this.cards = container.querySelectorAll('.video-card');
        this.dotsContainer = container.querySelector('.carousel-dots');
        this.currentIndex = 0;
        this.cardsPerView = this.getCardsPerView();
        this.totalSlides = Math.ceil(this.cards.length / this.cardsPerView);

        this.init();
    }

    init() {
        this.createDots();
        this.dots = this.dotsContainer.querySelectorAll('.carousel-dot');
        this.updateDots();

        // Update on resize
        window.addEventListener('resize', () => {
            const newCardsPerView = this.getCardsPerView();
            if (newCardsPerView !== this.cardsPerView) {
                this.cardsPerView = newCardsPerView;
                this.totalSlides = Math.ceil(this.cards.length / this.cardsPerView);
                this.currentIndex = Math.min(this.currentIndex, this.totalSlides - 1);
                this.createDots();
                this.dots = this.dotsContainer.querySelectorAll('.carousel-dot');
                this.goToSlide(this.currentIndex);
            }
        });

        // Touch/drag support for track
        let isDown = false;
        let startX;
        let scrollLeft;

        this.track.addEventListener('mousedown', (e) => {
            isDown = true;
            startX = e.pageX - this.track.offsetLeft;
            scrollLeft = this.track.scrollLeft;
        });

        this.track.addEventListener('mouseleave', () => {
            isDown = false;
        });

        this.track.addEventListener('mouseup', () => {
            isDown = false;
            this.snapToNearestSlide();
        });

        this.track.addEventListener('mousemove', (e) => {
            if (!isDown) return;
            e.preventDefault();
            const x = e.pageX - this.track.offsetLeft;
            const walk = (x - startX) * 2;
            this.track.scrollLeft = scrollLeft - walk;
        });

        // Touch events
        this.track.addEventListener('touchstart', (e) => {
            startX = e.touches[0].pageX - this.track.offsetLeft;
            scrollLeft = this.track.scrollLeft;
        }, { passive: true });

        this.track.addEventListener('touchend', () => {
            this.snapToNearestSlide();
        }, { passive: true });

        this.track.addEventListener('touchmove', (e) => {
            const x = e.touches[0].pageX - this.track.offsetLeft;
            const walk = (x - startX) * 2;
            this.track.scrollLeft = scrollLeft - walk;
        }, { passive: true });

        // Scroll listener to update dots
        this.track.addEventListener('scroll', () => {
            this.updateCurrentIndexFromScroll();
        }, { passive: true });
    }

    getCardsPerView() {
        const width = window.innerWidth;
        if (width <= 600) return 1;
        if (width <= 900) return 2;
        return 3;
    }

    createDots() {
        this.dotsContainer.innerHTML = '';
        for (let i = 0; i < this.totalSlides; i++) {
            const dot = document.createElement('button');
            dot.className = 'carousel-dot';
            dot.setAttribute('aria-label', `Go to slide ${i + 1}`);
            dot.addEventListener('click', () => this.goToSlide(i));
            this.dotsContainer.appendChild(dot);
        }
    }

    goToSlide(index) {
        this.currentIndex = index;
        const cardWidth = this.cards[0].offsetWidth;
        const gap = parseInt(getComputedStyle(this.track).gap) || 20;
        const scrollPosition = index * (cardWidth + gap) * this.cardsPerView;
        this.track.scrollTo({ left: scrollPosition, behavior: 'smooth' });
        this.updateDots();
    }

    snapToNearestSlide() {
        const cardWidth = this.cards[0].offsetWidth;
        const gap = parseInt(getComputedStyle(this.track).gap) || 20;
        const slideWidth = (cardWidth + gap) * this.cardsPerView;
        const nearestSlide = Math.round(this.track.scrollLeft / slideWidth);
        this.goToSlide(Math.max(0, Math.min(nearestSlide, this.totalSlides - 1)));
    }

    updateCurrentIndexFromScroll() {
        const cardWidth = this.cards[0].offsetWidth;
        const gap = parseInt(getComputedStyle(this.track).gap) || 20;
        const slideWidth = (cardWidth + gap) * this.cardsPerView;
        const newIndex = Math.round(this.track.scrollLeft / slideWidth);
        if (newIndex !== this.currentIndex && newIndex >= 0 && newIndex < this.totalSlides) {
            this.currentIndex = newIndex;
            this.updateDots();
        }
    }

    updateDots() {
        this.dots.forEach((dot, index) => {
            dot.classList.toggle('active', index === this.currentIndex);
        });
    }
}
