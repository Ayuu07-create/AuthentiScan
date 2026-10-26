(function () {
    "use strict";

    const menuToggle = document.getElementById("menuToggle");
    const sidebar = document.getElementById("appSidebar");
    const scrim = document.getElementById("sidebarScrim");

    function setSidebar(open) {
        if (!sidebar || !scrim || !menuToggle) return;
        sidebar.classList.toggle("open", open);
        scrim.classList.toggle("visible", open);
        menuToggle.setAttribute("aria-expanded", String(open));
        document.body.classList.toggle("nav-open", open);
    }

    if (menuToggle) {
        menuToggle.addEventListener("click", function () {
            setSidebar(!sidebar.classList.contains("open"));
        });
    }

    if (scrim) {
        scrim.addEventListener("click", function () { setSidebar(false); });
    }

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") setSidebar(false);
    });

    const fileInputs = document.querySelectorAll("input[type='file'][data-file-input]");
    fileInputs.forEach(function (input) {
        const target = document.getElementById(input.dataset.fileInput);
        input.addEventListener("change", function () {
            if (!target) return;
            const files = Array.from(input.files || []);
            target.textContent = files.length ? files.map(file => file.name).join(", ") : "No file selected";
            target.classList.toggle("has-file", files.length > 0);
        });
    });

    const forms = document.querySelectorAll("form[data-loading-form]");
    forms.forEach(function (form) {
        form.addEventListener("submit", function () {
            const button = form.querySelector("button[type='submit']");
            if (!button || button.disabled) return;
            button.dataset.originalText = button.innerHTML;
            button.innerHTML = "<span class='button-spinner'></span> Processing…";
            button.disabled = true;
            button.classList.add("is-loading");
        });
    });

    const passwordToggles = document.querySelectorAll("[data-password-toggle]");
    passwordToggles.forEach(function (button) {
        button.addEventListener("click", function () {
            const input = document.getElementById(button.dataset.passwordToggle);
            if (!input) return;
            const showing = input.type === "text";
            input.type = showing ? "password" : "text";
            button.textContent = showing ? "Show" : "Hide";
        });
    });
})();
