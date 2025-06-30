// TypeScript for CineVerse frontend

document.addEventListener('DOMContentLoaded', () => {
    // View switching functionality
    const islandButtons = document.querySelectorAll<HTMLDivElement>('.island-btn');
    const views = document.querySelectorAll<HTMLDivElement>('.view');

    islandButtons.forEach(button => {
        button.addEventListener('click', function () {
            const targetView = (this as HTMLElement).getAttribute('data-view');
            if (!targetView) return;

            // Update button states
            islandButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Update view visibility
            views.forEach(view => {
                view.classList.remove('active');
            });

            const target = document.getElementById(targetView);
            if (target) target.classList.add('active');
        });
    });

    // Simulate video placeholder animation
    const reelVideo = document.querySelector<HTMLDivElement>('.reel-video');
    const colors = ['#2d2b55', '#3e3b73', '#1a1838', '#4a2c6d'];
    let currentColor = 0;

    if (reelVideo) {
        setInterval(() => {
            reelVideo.style.background = `linear-gradient(45deg, ${colors[currentColor]}, ${colors[(currentColor + 1) % colors.length]})`;
            currentColor = (currentColor + 1) % colors.length;
        }, 3000);
    }

    // Add hover effects to movie cards
    const movieCards = document.querySelectorAll<HTMLDivElement>('.movie-card');
    movieCards.forEach(card => {
        card.addEventListener('mouseenter', function () {
            (this as HTMLElement).style.transform = 'translateY(-5px)';
        });

        card.addEventListener('mouseleave', function () {
            (this as HTMLElement).style.transform = 'translateY(0)';
        });
    });
});
