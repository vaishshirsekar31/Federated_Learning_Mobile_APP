const presets = {
    quick: {
        algorithm: "fedavg",
        rounds: 3,
        num_clients: 4,
        client_fraction: 0.5,
        local_epochs: 1,
        batch_size: 64,
        lr: 0.001,
        alpha: 0.5,
        mu: 0.01,
        seed: 42
    },
    balanced: {
        algorithm: "fedavg",
        rounds: 5,
        num_clients: 6,
        client_fraction: 0.5,
        local_epochs: 1,
        batch_size: 64,
        lr: 0.001,
        alpha: 0.3,
        mu: 0.01,
        seed: 42
    },
    strong: {
        algorithm: "fedprox",
        rounds: 10,
        num_clients: 8,
        client_fraction: 0.6,
        local_epochs: 2,
        batch_size: 64,
        lr: 0.001,
        alpha: 0.2,
        mu: 0.02,
        seed: 42
    }
};

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function syncRangeValues() {
    const clientFraction = document.getElementById("client_fraction");
    const alpha = document.getElementById("alpha");
    const mu = document.getElementById("mu");

    setText("client_fraction_value", clientFraction.value);
    setText("alpha_value", alpha.value);
    setText("mu_value", mu.value);
}

function toggleMuState() {
    const algorithm = document.getElementById("algorithm").value;
    const mu = document.getElementById("mu");
    const muHelp = document.getElementById("muHelp");

    const isFedProx = algorithm === "fedprox";
    mu.disabled = !isFedProx;
    mu.style.opacity = isFedProx ? "1" : "0.5";
    muHelp.textContent = isFedProx
        ? "Mu is active because FedProx is selected."
        : "Mu is ignored when FedAvg is selected.";
}

function applyPreset(presetName) {
    const preset = presets[presetName];
    if (!preset) return;

    document.getElementById("algorithm").value = preset.algorithm;
    document.getElementById("rounds").value = preset.rounds;
    document.getElementById("num_clients").value = preset.num_clients;
    document.getElementById("client_fraction").value = preset.client_fraction;
    document.getElementById("local_epochs").value = preset.local_epochs;
    document.getElementById("batch_size").value = preset.batch_size;
    document.getElementById("lr").value = preset.lr;
    document.getElementById("alpha").value = preset.alpha;
    document.getElementById("mu").value = preset.mu;
    document.getElementById("seed").value = preset.seed;

    syncRangeValues();
    toggleMuState();
}

function resetFormToDefaults() {
    applyPreset("balanced");
}

function enableLoadingState() {
    const overlay = document.getElementById("loadingOverlay");
    const btn = document.getElementById("trainBtn");

    overlay.classList.remove("hidden");
    btn.disabled = true;
    btn.textContent = "Training...";
}

function filterTable() {
    const input = document.getElementById("clientSearch");
    const table = document.getElementById("clientTable");
    if (!input || !table) return;

    const query = input.value.toLowerCase();
    const rows = table.querySelectorAll("tbody tr");

    rows.forEach((row) => {
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? "" : "none";
    });
}

function sortTableByAccuracy() {
    const table = document.getElementById("clientTable");
    if (!table) return;

    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));

    rows.sort((a, b) => {
        const accA = parseFloat(a.cells[2].innerText);
        const accB = parseFloat(b.cells[2].innerText);
        return accB - accA;
    });

    rows.forEach((row) => tbody.appendChild(row));
}

function scrollToResults() {
    const results = document.getElementById("results");
    if (results) {
        results.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    syncRangeValues();
    toggleMuState();

    ["client_fraction", "alpha", "mu"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("input", syncRangeValues);
    });

    const algorithm = document.getElementById("algorithm");
    if (algorithm) algorithm.addEventListener("change", toggleMuState);

    document.querySelectorAll(".preset-btn[data-preset]").forEach((btn) => {
        btn.addEventListener("click", () => {
            applyPreset(btn.dataset.preset);
        });
    });

    const resetBtn = document.getElementById("resetFormBtn");
    if (resetBtn) resetBtn.addEventListener("click", resetFormToDefaults);

    const form = document.getElementById("trainForm");
    if (form) {
        form.addEventListener("submit", () => {
            enableLoadingState();
        });
    }

    const searchInput = document.getElementById("clientSearch");
    if (searchInput) searchInput.addEventListener("input", filterTable);

    const sortBtn = document.getElementById("sortAccuracyBtn");
    if (sortBtn) sortBtn.addEventListener("click", sortTableByAccuracy);

    const scrollBtn = document.getElementById("scrollResultsBtn");
    if (scrollBtn) scrollBtn.addEventListener("click", scrollToResults);
});