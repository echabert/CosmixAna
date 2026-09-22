(function () {
    function startProgress() {
        const container = document.getElementById("measurement-progress-container");
        const bar = document.getElementById("measurement-progress-bar");
        const label = document.getElementById("measurement-progress-label");
        const durationInput = document.getElementById("duration");
        if (!container || !bar || !label) return;

        const duration = Math.max(1, Number(durationInput && durationInput.value) || 10);
        const isEnglish = document.documentElement.lang === "en";
        label.textContent = isEnglish ? "Measurement progress: 0%" : "Progression de la mesure : 0%";
        container.style.maxHeight = "100px";
        container.style.opacity = "1";
        container.style.margin = "16px 24px";
        bar.style.transition = "none";
        bar.style.width = "0%";
        window.requestAnimationFrame(function () {
            bar.style.transition = "width " + duration + "s linear";
            bar.style.width = "100%";
        });
    }

    document.addEventListener("click", function (event) {
        if (event.target.closest && event.target.closest("#run-button")) {
            startProgress();
        }
    });

})();