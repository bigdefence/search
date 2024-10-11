document.addEventListener('DOMContentLoaded', function() {
    // 이미지 갤러리 모달
    const imageGallery = document.querySelector('.image-gallery');
    if (imageGallery) {
        imageGallery.addEventListener('click', function(e) {
            if (e.target.tagName === 'IMG') {
                const modal = document.createElement('div');
                modal.className = 'image-modal';
                modal.innerHTML = `
                    <div class="modal-content">
                        <img src="${e.target.src}" alt="${e.target.alt}">
                        <p>${e.target.alt}</p>
                    </div>
                `;
                document.body.appendChild(modal);
                modal.addEventListener('click', function() {
                    document.body.removeChild(modal);
                });
            }
        });
    }

    // 코드 블록 복사 기능
    document.querySelectorAll('pre code').forEach((block) => {
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.textContent = '복사';
        block.parentNode.insertBefore(copyButton, block);

        copyButton.addEventListener('click', function() {
            const code = block.textContent;
            navigator.clipboard.writeText(code).then(() => {
                copyButton.textContent = '복사됨!';
                setTimeout(() => {
                    copyButton.textContent = '복사';
                }, 2000);
            });
        });
    });

    // 검색 결과 링크 클릭 추적
    document.querySelectorAll('.result-item a').forEach(link => {
        link.addEventListener('click', function(e) {
            const url = this.href;
            fetch('/record_click', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });
        });
    });

    // 비디오 썸네일 오류 처리
    document.querySelectorAll('.video-thumbnail').forEach(img => {
        img.addEventListener('error', function() {
            this.onerror = null;
            this.src = '/static/placeholder.png';
        });
    });

    // 모바일에서 폼 제출 시 키보드 닫기
    const chatInput = document.querySelector('.chat-input-container input[type="text"]');
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.form.dispatchEvent(new Event('submit'));
            this.blur();
        }
    });

    // 모바일에서 스크롤 개선
    function smoothScroll(element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }

    form.addEventListener('submit', function(e) {
        // 기존 제출 코드...

        // 폼 제출 후 결과 영역으로 부드럽게 스크롤
        setTimeout(() => {
            const resultsContainer = document.querySelector('.results-container');
            if (resultsContainer) {
                smoothScroll(resultsContainer);
            }
        }, 100);
    });

    // 이미지 모달 닫기 버튼 추가 (모바일에서 더 쉽게 닫을 수 있도록)
    function setupImageGallery() {
        const imageGallery = document.querySelector('.image-gallery');
        if (imageGallery) {
            imageGallery.addEventListener('click', function(e) {
                if (e.target.tagName === 'IMG') {
                    const modal = document.createElement('div');
                    modal.className = 'image-modal';
                    modal.innerHTML = `
                        <div class="modal-content">
                            <span class="close-modal">&times;</span>
                            <img src="${e.target.src}" alt="${e.target.alt}">
                            <p>${e.target.alt}</p>
                        </div>
                    `;
                    document.body.appendChild(modal);
                    
                    const closeButton = modal.querySelector('.close-modal');
                    closeButton.addEventListener('click', function() {
                        document.body.removeChild(modal);
                    });
                    
                    modal.addEventListener('click', function(e) {
                        if (e.target === modal) {
                            document.body.removeChild(modal);
                        }
                    });
                }
            });
        }
    }

    // 기존 코드 유지
});