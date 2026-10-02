(function () {
    document.addEventListener("click", async function (event) {
        const button = event.target.closest && event.target.closest("#export-page-button");
        if (!button || !window.Plotly) return;

        const page = window.location.pathname.replace(/^\//, "");
        if (!["poisson", "trends", "angle"].includes(page)) return;

        const charts = Array.from(document.querySelectorAll("#page-content .js-plotly-plot"))
            .filter(function (chart) { return chart.data && chart.data.length; });
        const status = document.getElementById("export-status");
        const selectedLanguage = document.querySelector("#lang-select .Select-value-label");
        const isEnglish = /english/i.test(selectedLanguage && selectedLanguage.textContent || "");

        if (charts.length === 0) {
            if (status) status.textContent = isEnglish ? "No charts available to export." : "Aucun graphique à exporter.";
            return;
        }

        button.disabled = true;
        if (status) status.textContent = isEnglish ? "Exporting…" : "Export en cours…";

        try {
            const images = await Promise.all(charts.map(async function (chart) {
                return {
                    data: await window.Plotly.toImage(chart, {
                        format: "png",
                        width: 1400,
                        height: 850,
                        scale: 1,
                    }),
                };
            }));
            const response = await fetch("/api/export-page-images", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ page: page, images: images }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || "Image export failed");
            if (status) {
                const saved = isEnglish
                    ? "Saved " + result.files.length + " images in exports/"
                    : result.files.length + " images enregistrées dans exports/";
                if (result.report) {
                    status.textContent = isEnglish
                        ? saved + " — report updated (exports/" + result.report + ")"
                        : saved + " — rapport mis à jour (exports/" + result.report + ")";
                } else {
                    status.textContent = isEnglish
                        ? saved + " — HTML report not updated."
                        : saved + " — rapport HTML non mis à jour.";
                }
            }
        } catch (error) {
            if (status) status.textContent = isEnglish ? "Image export failed." : "Échec de l'export des images.";
        } finally {
            button.disabled = false;
        }
    });
})();