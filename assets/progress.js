(function () {
    let progressFrame;

    function startProgress() {
        const container = document.getElementById("measurement-progress-container");
        const bar = document.getElementById("measurement-progress-bar");
        const label = document.getElementById("measurement-progress-label");
        const durationInput = document.getElementById("duration");
        if (!container || !bar || !label) return;

        const duration = Math.max(1, Number(durationInput && durationInput.value) || 60);
        const isEnglish = document.documentElement.lang === "en";
        const progressText = isEnglish ? "Measurement progress" : "Progression de la mesure";
        if (progressFrame !== undefined) {
            window.cancelAnimationFrame(progressFrame);
        }
        label.textContent = progressText + ": 0%";
        container.style.maxHeight = "100px";
        container.style.opacity = "1";
        container.style.margin = "16px 24px";
        bar.style.transition = "none";
        bar.style.width = "0%";
        const startTime = performance.now();

        function updateProgress(timestamp) {
            const progress = Math.min(100, Math.floor(((timestamp - startTime) / (duration * 1000)) * 100));
            label.textContent = progressText + ": " + progress + "%";
            bar.style.width = progress + "%";
            if (progress < 100) {
                progressFrame = window.requestAnimationFrame(updateProgress);
            }
        }

        window.requestAnimationFrame(function () {
            progressFrame = window.requestAnimationFrame(updateProgress);
        });
    }

    document.addEventListener("click", function (event) {
        if (event.target.closest && event.target.closest("#run-button")) {
            startProgress();
        }
    });

})();