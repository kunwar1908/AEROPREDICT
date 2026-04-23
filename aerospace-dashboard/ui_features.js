(() => {
  const DEFAULT_API_BASE = "http://127.0.0.1:8000";
  const SETTINGS_KEY = "aeropredict-ui-settings";
  const NOTIFICATION_DISMISS_KEY = "aeropredict-notifications-dismissed-at";
  const DEFAULT_SETTINGS = {
    apiBase: DEFAULT_API_BASE,
    autoRefreshSeconds: 45,
    compactTables: false,
    reduceMotion: false,
    toastNotifications: true,
  };
  const TIME_RANGES = ["Last 24 Hours", "Last 7 Days", "Last 30 Days"];
  let metricsIntervalId = null;
  let notificationsIntervalId = null;
  let unreadNotificationCount = 0;
  let latestNotifications = [];
  const searchState = { query: "", matches: [], index: -1 };
  let uiSettings = loadUiSettings();

  function sanitizeSettings(candidate) {
    const source = candidate && typeof candidate === "object" ? candidate : {};
    const apiBase = String(source.apiBase || DEFAULT_API_BASE).trim().replace(/\/$/, "") || DEFAULT_API_BASE;
    const parsedSeconds = Number(source.autoRefreshSeconds);
    const autoRefreshSeconds = Number.isFinite(parsedSeconds)
      ? Math.min(300, Math.max(15, Math.round(parsedSeconds)))
      : DEFAULT_SETTINGS.autoRefreshSeconds;

    return {
      apiBase,
      autoRefreshSeconds,
      compactTables: Boolean(source.compactTables),
      reduceMotion: Boolean(source.reduceMotion),
      toastNotifications: source.toastNotifications !== false,
    };
  }

  function loadUiSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return { ...DEFAULT_SETTINGS };
      return sanitizeSettings(JSON.parse(raw));
    } catch {
      return { ...DEFAULT_SETTINGS };
    }
  }

  function saveUiSettings(next) {
    uiSettings = sanitizeSettings(next);
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(uiSettings));
    applyUiSettings();
    window.dispatchEvent(new CustomEvent("aeropredict:settings-updated", { detail: { ...uiSettings } }));
  }

  function getApiBase() {
    return uiSettings.apiBase || DEFAULT_API_BASE;
  }

  function restartMetricsInterval() {
    if (metricsIntervalId != null) {
      window.clearInterval(metricsIntervalId);
      metricsIntervalId = null;
    }

    if (document.body && document.body.dataset.floatingMetrics === "true") {
      metricsIntervalId = window.setInterval(() => refreshMetricsDock(false), uiSettings.autoRefreshSeconds * 1000);
    }
  }

  function applyUiSettings() {
    document.body.classList.toggle("aeropredict-compact", Boolean(uiSettings.compactTables));
    document.body.classList.toggle("aeropredict-reduce-motion", Boolean(uiSettings.reduceMotion));
    restartMetricsInterval();
  }

  function ensureUiFeatureStyles() {
    if (document.getElementById("aeropredict-ui-feature-style")) return;

    const style = document.createElement("style");
    style.id = "aeropredict-ui-feature-style";
    style.textContent = `
      #aeropredict-settings-modal {
        position: fixed;
        inset: 0;
        z-index: 75;
        display: none;
        align-items: center;
        justify-content: center;
        background: rgba(2, 6, 23, 0.52);
        backdrop-filter: blur(3px);
      }
      #aeropredict-settings-panel {
        width: min(92vw, 560px);
        max-height: min(86vh, 720px);
        overflow: auto;
        border-radius: 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.36);
        background: rgba(9, 13, 28, 0.96);
        color: #f8fafc;
        padding: 1rem;
        box-shadow: 0 20px 48px rgba(2, 6, 23, 0.38);
      }
      html[data-theme="light"] #aeropredict-settings-panel {
        background: rgba(255, 255, 255, 0.98);
        color: #0f172a;
        border-color: rgba(148, 163, 184, 0.52);
      }
      .aeropredict-setting-field {
        display: grid;
        gap: 0.38rem;
        margin-bottom: 0.8rem;
      }
      .aeropredict-setting-field label {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
      }
      html[data-theme="light"] .aeropredict-setting-field label {
        color: #475569;
      }
      .aeropredict-setting-field input[type="text"],
      .aeropredict-setting-field input[type="number"] {
        width: 100%;
        border-radius: 0.62rem;
        border: 1px solid rgba(148, 163, 184, 0.36);
        background: rgba(15, 23, 42, 0.52);
        color: #f8fafc;
        padding: 0.56rem 0.7rem;
        font-size: 0.84rem;
      }
      html[data-theme="light"] .aeropredict-setting-field input[type="text"],
      html[data-theme="light"] .aeropredict-setting-field input[type="number"] {
        background: #ffffff;
        color: #0f172a;
      }
      .aeropredict-switch-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.85rem;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 0.62rem;
        padding: 0.56rem 0.7rem;
        margin-bottom: 0.6rem;
      }
      .aeropredict-switch-row p {
        margin: 0;
        font-size: 0.72rem;
        color: #94a3b8;
      }
      html[data-theme="light"] .aeropredict-switch-row p {
        color: #475569;
      }
      .aeropredict-settings-actions {
        display: flex;
        justify-content: flex-end;
        gap: 0.55rem;
        margin-top: 0.6rem;
      }
      .aeropredict-settings-actions button {
        border-radius: 0.6rem;
        border: 1px solid rgba(148, 163, 184, 0.34);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.45rem 0.62rem;
      }
      .aeropredict-settings-actions .primary {
        border-color: rgba(249, 115, 22, 0.42);
        background: #ea580c;
        color: #ffffff;
      }
      body.aeropredict-compact table th,
      body.aeropredict-compact table td {
        padding-top: 0.45rem !important;
        padding-bottom: 0.45rem !important;
      }
      body.aeropredict-reduce-motion *,
      body.aeropredict-reduce-motion *::before,
      body.aeropredict-reduce-motion *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
      }

      #aeropredict-notify-panel {
        position: fixed;
        top: 0;
        right: 0;
        height: 100vh;
        width: min(92vw, 360px);
        z-index: 74;
        transform: translateX(105%);
        transition: transform 220ms ease;
        border-left: 1px solid rgba(148, 163, 184, 0.34);
        background: rgba(9, 13, 28, 0.98);
        color: #f8fafc;
        box-shadow: -18px 0 40px rgba(2, 6, 23, 0.4);
        display: flex;
        flex-direction: column;
      }
      #aeropredict-notify-panel.open {
        transform: translateX(0);
      }
      #aeropredict-notify-backdrop {
        position: fixed;
        inset: 0;
        z-index: 73;
        background: rgba(2, 6, 23, 0.36);
        backdrop-filter: blur(2px);
        display: none;
      }
      #aeropredict-notify-backdrop.open {
        display: block;
      }
      #aeropredict-notify-panel header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        padding: 0.9rem 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.24);
      }
      #aeropredict-notify-panel h3 {
        margin: 0;
        font-size: 0.8rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }
      #aeropredict-notify-panel .notify-close {
        border: 1px solid rgba(148, 163, 184, 0.34);
        border-radius: 0.55rem;
        background: transparent;
        color: inherit;
        padding: 0.28rem 0.5rem;
        font-size: 0.72rem;
        cursor: pointer;
      }
      #aeropredict-notify-content {
        padding: 0.9rem 1rem;
        overflow: auto;
      }
      .aeropredict-notify-card {
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-radius: 0.7rem;
        padding: 0.85rem;
        background: rgba(15, 23, 42, 0.5);
      }
      .aeropredict-notify-card-title {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #fb923c;
        margin-bottom: 0.35rem;
      }
      .aeropredict-notify-card-text {
        margin: 0;
        font-size: 0.82rem;
        color: #cbd5e1;
      }
      .aeropredict-notify-meta {
        margin-top: 0.55rem;
        font-size: 0.68rem;
        letter-spacing: 0.03em;
        color: #94a3b8;
      }
      html[data-theme="light"] #aeropredict-notify-panel {
        background: rgba(255, 255, 255, 0.98);
        color: #0f172a;
        border-left-color: rgba(148, 163, 184, 0.48);
      }
      html[data-theme="light"] .aeropredict-notify-card {
        background: #ffffff;
        border-color: rgba(148, 163, 184, 0.48);
      }
      html[data-theme="light"] .aeropredict-notify-card-text {
        color: #334155;
      }
      html[data-theme="light"] .aeropredict-notify-meta {
        color: #64748b;
      }

      .aeropredict-notify-badge {
        position: absolute;
        top: -0.3rem;
        right: -0.35rem;
        min-width: 1.05rem;
        height: 1.05rem;
        border-radius: 999px;
        border: 1px solid rgba(100, 116, 139, 0.45);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        background: rgba(15, 23, 42, 0.95);
        color: #cbd5e1;
      }

      .aeropredict-notify-badge.has-unread {
        border-color: rgba(249, 115, 22, 0.45);
        background: rgba(194, 65, 12, 0.9);
        color: #ffffff;
      }

      html[data-theme="light"] .aeropredict-notify-badge {
        background: rgba(255, 255, 255, 0.98);
        color: #475569;
        border-color: rgba(148, 163, 184, 0.6);
      }

      html[data-theme="light"] .aeropredict-notify-badge.has-unread {
        background: rgba(234, 88, 12, 0.95);
        color: #ffffff;
        border-color: rgba(194, 65, 12, 0.65);
      }

      .aeropredict-search-hit {
        outline: 2px solid rgba(249, 115, 22, 0.48);
        border-radius: 0.3rem;
        transition: outline-color 160ms ease;
      }

      .aeropredict-search-hit-active {
        outline-color: rgba(249, 115, 22, 0.96);
        box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.2);
      }
    `;
    document.head.appendChild(style);
  }

  function clearSearchHighlights() {
    searchState.matches.forEach((node) => {
      node.classList.remove("aeropredict-search-hit", "aeropredict-search-hit-active");
    });
    searchState.matches = [];
    searchState.index = -1;
  }

  function getSearchableNodes() {
    const scope = document.querySelector("main") || document.body;
    const selector = [
      "h1",
      "h2",
      "h3",
      "h4",
      "p",
      "a",
      "button",
      "label",
      "td",
      "th",
      "span",
      "li",
      ".code-font pre",
    ].join(",");
    const nodes = Array.from(scope.querySelectorAll(selector));
    return nodes.filter((node) => {
      if (!(node instanceof HTMLElement)) return false;
      if (!node.textContent || !node.textContent.trim()) return false;
      if (node.closest("script, style, noscript")) return false;
      const rect = node.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    });
  }

  function focusSearchMatch(index) {
    if (!searchState.matches.length) return;
    const safeIndex = ((index % searchState.matches.length) + searchState.matches.length) % searchState.matches.length;
    searchState.index = safeIndex;

    searchState.matches.forEach((node, i) => {
      node.classList.toggle("aeropredict-search-hit-active", i === safeIndex);
    });

    const active = searchState.matches[safeIndex];
    active.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
  }

  function runSearchQuery(rawQuery) {
    const query = normalizeText(rawQuery);
    if (!query) {
      clearSearchHighlights();
      showToast("Type a search term and press Enter.", "info");
      return;
    }

    if (query === searchState.query && searchState.matches.length > 0) {
      focusSearchMatch(searchState.index + 1);
      showToast(`Match ${searchState.index + 1} of ${searchState.matches.length}.`, "info");
      return;
    }

    clearSearchHighlights();
    const allNodes = getSearchableNodes();
    const matches = allNodes.filter((node) => normalizeText(node.textContent).includes(query));

    if (!matches.length) {
      searchState.query = query;
      showToast("No matches found on this page.", "warning");
      return;
    }

    searchState.query = query;
    searchState.matches = matches.slice(0, 150);
    searchState.matches.forEach((node) => node.classList.add("aeropredict-search-hit"));
    focusSearchMatch(0);
    showToast(`Found ${searchState.matches.length} matches. Press Enter again for next.`, "success");
  }

  function wireSearchInputs() {
    const candidates = Array.from(document.querySelectorAll('input[type="search"], input[type="text"]')).filter((input) => {
      const placeholder = normalizeText(input.getAttribute("placeholder"));
      return placeholder.includes("search");
    });

    candidates.forEach((input) => {
      if (input.dataset.searchWired === "true") return;
      input.dataset.searchWired = "true";

      input.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        runSearchQuery(input.value);
      });

      input.addEventListener("input", () => {
        if (!input.value.trim()) {
          clearSearchHighlights();
          searchState.query = "";
        }
      });
    });
  }

  function ensureNotificationBadges() {
    document.querySelectorAll('[data-icon="notifications"]').forEach((iconNode) => {
      const host = iconNode.closest("button") || iconNode;
      if (!(host instanceof HTMLElement)) return;
      if (host.dataset.notifyBadgeWired !== "true") {
        host.dataset.notifyBadgeWired = "true";
        host.style.position = "relative";
      }

      let badge = host.querySelector(".aeropredict-notify-badge");
      if (!badge) {
        badge = document.createElement("span");
        badge.className = "aeropredict-notify-badge";
        badge.setAttribute("aria-label", "Unread notifications");
        host.appendChild(badge);
      }

      badge.textContent = String(unreadNotificationCount);
      badge.classList.toggle("has-unread", unreadNotificationCount > 0);
    });
  }

  function setUnreadNotifications(count) {
    const parsed = Number(count);
    unreadNotificationCount = Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : 0;
    ensureNotificationBadges();
  }

  function getDismissedNotificationsAt() {
    const raw = localStorage.getItem(NOTIFICATION_DISMISS_KEY);
    const parsed = raw ? Date.parse(raw) : NaN;
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function dismissNotifications() {
    localStorage.setItem(NOTIFICATION_DISMISS_KEY, new Date().toISOString());
    latestNotifications = [];
    setUnreadNotifications(0);
    renderNotificationsInPanel([]);
    closeNotificationsPanel();
    showToast("Notifications dismissed.", "info", true);
  }

  function getNotificationEndpoint() {
    return `${getApiBase()}/api/notifications`;
  }

  function renderNotificationsInPanel(notifications) {
    const panel = document.getElementById("aeropredict-notify-panel");
    if (!panel) return;
    const content = panel.querySelector("#aeropredict-notify-content");
    if (!content) return;

    const items = Array.isArray(notifications) ? notifications : [];
    if (!items.length) {
      content.innerHTML = `
        <div class="aeropredict-notify-card">
          <div class="aeropredict-notify-card-title">No New Updates</div>
          <p class="aeropredict-notify-card-text">No new alerts in queue.</p>
          <div class="aeropredict-notify-meta" id="aeropredict-notify-meta">Checked ${new Date().toLocaleTimeString()}</div>
        </div>
      `;
      return;
    }

    content.innerHTML = items
      .map((item) => {
        const title = String(item.title || "Notification");
        const message = String(item.message || "");
        const level = String(item.level || "info").toUpperCase();
        const createdAt = item.createdAt ? new Date(String(item.createdAt)).toLocaleTimeString() : new Date().toLocaleTimeString();
        return `
          <div class="aeropredict-notify-card" style="margin-bottom:0.7rem;">
            <div class="aeropredict-notify-card-title">${title}</div>
            <p class="aeropredict-notify-card-text">${message}</p>
            <div class="aeropredict-notify-meta">${level} • ${createdAt}</div>
          </div>
        `;
      })
      .join("");
  }

  async function refreshNotifications(showFeedback = false) {
    try {
      const response = await fetch(getNotificationEndpoint(), { headers: { Accept: "application/json" } });
      if (!response.ok) {
        throw new Error(`notifications request failed (${response.status})`);
      }
      const payload = await response.json();
      const dismissedAt = getDismissedNotificationsAt();
      latestNotifications = (Array.isArray(payload.notifications) ? payload.notifications : []).filter((item) => {
        if (!item || !item.createdAt) return true;
        const createdAt = Date.parse(item.createdAt);
        return !Number.isFinite(createdAt) || createdAt > dismissedAt;
      });
      setUnreadNotifications(latestNotifications.length);
      renderNotificationsInPanel(latestNotifications);
      if (showFeedback) showToast("Notifications refreshed.", "success");
    } catch {
      latestNotifications = [];
      setUnreadNotifications(0);
      renderNotificationsInPanel([]);
      if (showFeedback) showToast("Notifications unavailable. API may be offline.", "warning");
    }
  }

  function startNotificationsPolling() {
    if (notificationsIntervalId != null) {
      window.clearInterval(notificationsIntervalId);
      notificationsIntervalId = null;
    }
    notificationsIntervalId = window.setInterval(() => {
      refreshNotifications(false);
    }, 30000);
  }

  function ensureNotificationsPanel() {
    let backdrop = document.getElementById("aeropredict-notify-backdrop");
    let panel = document.getElementById("aeropredict-notify-panel");
    if (backdrop && panel) return { backdrop, panel };

    ensureUiFeatureStyles();

    backdrop = document.createElement("div");
    backdrop.id = "aeropredict-notify-backdrop";

    panel = document.createElement("aside");
    panel.id = "aeropredict-notify-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-label", "Notifications");
    panel.innerHTML = `
      <header>
        <h3>Notifications</h3>
        <div style="display:flex;gap:0.45rem;align-items:center;">
          <button type="button" class="notify-close" id="aeropredict-notify-dismiss">Dismiss</button>
          <button type="button" class="notify-close" id="aeropredict-notify-close">Close</button>
        </div>
      </header>
      <div id="aeropredict-notify-content">
        <div class="aeropredict-notify-card">
          <div class="aeropredict-notify-card-title">No New Updates</div>
          <p class="aeropredict-notify-card-text">No new alerts in queue.</p>
          <div class="aeropredict-notify-meta" id="aeropredict-notify-meta">Checked just now</div>
        </div>
      </div>
    `;

    document.body.appendChild(backdrop);
    document.body.appendChild(panel);

    const dismissBtn = panel.querySelector("#aeropredict-notify-dismiss");
    if (dismissBtn) dismissBtn.addEventListener("click", dismissNotifications);
    const closeBtn = panel.querySelector("#aeropredict-notify-close");
    if (closeBtn) closeBtn.addEventListener("click", closeNotificationsPanel);
    backdrop.addEventListener("click", closeNotificationsPanel);

    return { backdrop, panel };
  }

  async function openNotificationsPanel() {
    const { backdrop, panel } = ensureNotificationsPanel();
    await refreshNotifications(false);
    backdrop.classList.add("open");
    panel.classList.add("open");
  }

  function closeNotificationsPanel() {
    const backdrop = document.getElementById("aeropredict-notify-backdrop");
    const panel = document.getElementById("aeropredict-notify-panel");
    if (backdrop) backdrop.classList.remove("open");
    if (panel) panel.classList.remove("open");
  }

  function ensureSettingsModal() {
    let modal = document.getElementById("aeropredict-settings-modal");
    if (modal) return modal;

    ensureUiFeatureStyles();
    modal = document.createElement("div");
    modal.id = "aeropredict-settings-modal";
    modal.innerHTML = `
      <div id="aeropredict-settings-panel" role="dialog" aria-modal="true" aria-label="Dashboard Settings">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;margin-bottom:0.7rem;">
          <div>
            <h3 style="margin:0;font-size:0.95rem;font-weight:900;letter-spacing:0.06em;text-transform:uppercase;">Dashboard Settings</h3>
            <p style="margin:0.28rem 0 0 0;font-size:0.74rem;color:#94a3b8;">Control runtime preferences saved on this browser.</p>
          </div>
          <button id="aeropredict-settings-close" type="button" style="border:1px solid rgba(148,163,184,0.34);background:transparent;color:inherit;border-radius:0.55rem;padding:0.3rem 0.45rem;cursor:pointer;">Close</button>
        </div>

        <div class="aeropredict-setting-field">
          <label for="aeropredict-settings-api">API Base URL</label>
          <input id="aeropredict-settings-api" type="text" placeholder="http://127.0.0.1:8000" />
        </div>

        <div class="aeropredict-setting-field">
          <label for="aeropredict-settings-refresh">Metrics Auto Refresh (seconds)</label>
          <input id="aeropredict-settings-refresh" type="number" min="15" max="300" step="1" />
        </div>

        <div class="aeropredict-switch-row">
          <div>
            <div style="font-size:0.78rem;font-weight:700;">Compact tables</div>
            <p>Tighter row spacing for telemetry grids.</p>
          </div>
          <input id="aeropredict-settings-compact" type="checkbox" />
        </div>

        <div class="aeropredict-switch-row">
          <div>
            <div style="font-size:0.78rem;font-weight:700;">Reduce motion</div>
            <p>Minimizes transitions and animated indicators.</p>
          </div>
          <input id="aeropredict-settings-motion" type="checkbox" />
        </div>

        <div class="aeropredict-switch-row">
          <div>
            <div style="font-size:0.78rem;font-weight:700;">Toast notifications</div>
            <p>Shows operation feedback popups.</p>
          </div>
          <input id="aeropredict-settings-toasts" type="checkbox" />
        </div>

        <div class="aeropredict-settings-actions">
          <button id="aeropredict-settings-defaults" type="button">Reset defaults</button>
          <button id="aeropredict-settings-theme" type="button">Toggle theme</button>
          <button id="aeropredict-settings-save" class="primary" type="button">Save settings</button>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    modal.addEventListener("click", (event) => {
      if (event.target === modal) {
        closeSettingsModal();
      }
    });

    const closeBtn = modal.querySelector("#aeropredict-settings-close");
    if (closeBtn) closeBtn.addEventListener("click", closeSettingsModal);

    const defaultsBtn = modal.querySelector("#aeropredict-settings-defaults");
    if (defaultsBtn) {
      defaultsBtn.addEventListener("click", () => {
        saveUiSettings({ ...DEFAULT_SETTINGS });
        populateSettingsForm();
        showToast("Settings reset to defaults.", "info", true);
      });
    }

    const themeBtn = modal.querySelector("#aeropredict-settings-theme");
    if (themeBtn) {
      themeBtn.addEventListener("click", () => {
        if (window.AeroPredictTheme && typeof window.AeroPredictTheme.toggleTheme === "function") {
          window.AeroPredictTheme.toggleTheme();
        }
      });
    }

    const saveBtn = modal.querySelector("#aeropredict-settings-save");
    if (saveBtn) {
      saveBtn.addEventListener("click", () => {
        const apiInput = modal.querySelector("#aeropredict-settings-api");
        const refreshInput = modal.querySelector("#aeropredict-settings-refresh");
        const compactInput = modal.querySelector("#aeropredict-settings-compact");
        const motionInput = modal.querySelector("#aeropredict-settings-motion");
        const toastInput = modal.querySelector("#aeropredict-settings-toasts");

        const nextSettings = sanitizeSettings({
          apiBase: apiInput ? apiInput.value : uiSettings.apiBase,
          autoRefreshSeconds: refreshInput ? refreshInput.value : uiSettings.autoRefreshSeconds,
          compactTables: compactInput ? compactInput.checked : uiSettings.compactTables,
          reduceMotion: motionInput ? motionInput.checked : uiSettings.reduceMotion,
          toastNotifications: toastInput ? toastInput.checked : uiSettings.toastNotifications,
        });

        saveUiSettings(nextSettings);
        closeSettingsModal();
        showToast("Settings saved.", "success", true);
      });
    }

    return modal;
  }

  function populateSettingsForm() {
    const modal = ensureSettingsModal();
    const apiInput = modal.querySelector("#aeropredict-settings-api");
    const refreshInput = modal.querySelector("#aeropredict-settings-refresh");
    const compactInput = modal.querySelector("#aeropredict-settings-compact");
    const motionInput = modal.querySelector("#aeropredict-settings-motion");
    const toastInput = modal.querySelector("#aeropredict-settings-toasts");

    if (apiInput) apiInput.value = uiSettings.apiBase;
    if (refreshInput) refreshInput.value = String(uiSettings.autoRefreshSeconds);
    if (compactInput) compactInput.checked = uiSettings.compactTables;
    if (motionInput) motionInput.checked = uiSettings.reduceMotion;
    if (toastInput) toastInput.checked = uiSettings.toastNotifications;
  }

  function openSettingsModal() {
    const modal = ensureSettingsModal();
    populateSettingsForm();
    modal.style.display = "flex";
  }

  function closeSettingsModal() {
    const modal = document.getElementById("aeropredict-settings-modal");
    if (modal) modal.style.display = "none";
  }

  function ensureToastHost() {
    let host = document.getElementById("aeropredict-toast-host");
    if (host) return host;
    host = document.createElement("div");
    host.id = "aeropredict-toast-host";
    host.style.position = "fixed";
    host.style.top = "1rem";
    host.style.right = "1rem";
    host.style.zIndex = "70";
    host.style.display = "flex";
    host.style.flexDirection = "column";
    host.style.gap = "0.5rem";
    document.body.appendChild(host);
    return host;
  }

  function showToast(message, tone = "info", force = false) {
    if (!force && uiSettings.toastNotifications === false) return;
    const host = ensureToastHost();
    const toast = document.createElement("div");
    const toneMap = {
      info: { bg: "rgba(15,23,42,0.94)", border: "#334155" },
      success: { bg: "rgba(3,84,63,0.94)", border: "#2dd4bf" },
      warning: { bg: "rgba(120,53,15,0.94)", border: "#fb923c" },
      danger: { bg: "rgba(127,29,29,0.94)", border: "#f87171" },
    };
    const toneStyle = toneMap[tone] || toneMap.info;

    toast.style.background = toneStyle.bg;
    toast.style.border = `1px solid ${toneStyle.border}`;
    toast.style.color = "#f8fafc";
    toast.style.padding = "0.65rem 0.85rem";
    toast.style.borderRadius = "0.6rem";
    toast.style.boxShadow = "0 8px 24px rgba(2,6,23,0.28)";
    toast.style.fontSize = "0.78rem";
    toast.style.fontWeight = "600";
    toast.style.maxWidth = "320px";
    toast.textContent = message;

    host.appendChild(toast);
    setTimeout(() => {
      toast.remove();
      if (!host.childElementCount) host.remove();
    }, 2800);
  }

  function downloadJsonFile(name, payload) {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  async function fetchSummary() {
    const response = await fetch(`${getApiBase()}/api/summary`);
    if (!response.ok) throw new Error(`summary request failed (${response.status})`);
    return response.json();
  }

  function normalizeText(value) {
    return String(value || "").replace(/\s+/g, " ").trim().toLowerCase();
  }

  function ensureMetricsDock() {
    let dock = document.getElementById("aeropredict-metrics-dock");
    if (dock) return dock;

    dock = document.createElement("aside");
    dock.id = "aeropredict-metrics-dock";
    dock.style.position = "fixed";
    dock.style.right = "1rem";
    dock.style.top = "5rem";
    dock.style.zIndex = "65";
    dock.style.minWidth = "240px";
    dock.style.maxWidth = "280px";
    dock.style.padding = "0.7rem 0.75rem";
    dock.style.borderRadius = "0.8rem";
    dock.style.background = "rgba(9, 13, 28, 0.9)";
    dock.style.border = "1px solid rgba(100, 116, 139, 0.38)";
    dock.style.boxShadow = "0 10px 32px rgba(2,6,23,0.26)";
    dock.style.backdropFilter = "blur(6px)";
    dock.style.color = "#f8fafc";

    dock.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
        <div style="font-size:0.7rem;font-weight:800;letter-spacing:0.1em;text-transform:uppercase;color:#fb923c;">Model Metrics</div>
        <button id="aeropredict-metrics-refresh" type="button" style="font-size:0.66rem;border:1px solid rgba(148,163,184,0.4);background:transparent;color:#cbd5e1;padding:0.2rem 0.45rem;border-radius:999px;cursor:pointer;">Refresh</button>
      </div>
      <div id="aeropredict-metrics-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:0.35rem 0.5rem;font-size:0.73rem;"></div>
      <div id="aeropredict-metrics-foot" style="margin-top:0.5rem;font-size:0.64rem;color:#94a3b8;letter-spacing:0.04em;"></div>
    `;

    document.body.appendChild(dock);
    return dock;
  }

  function updateDockTheme(theme) {
    const dock = document.getElementById("aeropredict-metrics-dock");
    if (!dock) return;

    if (theme === "light") {
      dock.style.background = "rgba(255,255,255,0.95)";
      dock.style.border = "1px solid rgba(148,163,184,0.46)";
      dock.style.color = "#0f172a";
    } else {
      dock.style.background = "rgba(9, 13, 28, 0.9)";
      dock.style.border = "1px solid rgba(100, 116, 139, 0.38)";
      dock.style.color = "#f8fafc";
    }
  }

  function setMetrics(summary) {
    const grid = document.getElementById("aeropredict-metrics-grid");
    const foot = document.getElementById("aeropredict-metrics-foot");
    if (!grid || !foot) return;

    const metrics = summary?.metrics || {};
    const dataset = summary?.dataset || "--";
    const device = summary?.device || "--";
    const bestEpoch = metrics.bestEpoch ?? "--";
    const rmse = metrics.testRmse != null ? Number(metrics.testRmse).toFixed(2) : "--";
    const mae = metrics.testMae != null ? Number(metrics.testMae).toFixed(2) : "--";
    const nasa = metrics.nasaScore != null ? Number(metrics.nasaScore).toFixed(1) : "--";

    const labelColor = document.documentElement.dataset.theme === "light" ? "#475569" : "#94a3b8";
    const valueColor = document.documentElement.dataset.theme === "light" ? "#0f172a" : "#f8fafc";

    const cell = (label, value) => `
      <div style="padding:0.35rem 0.4rem;border:1px solid rgba(148,163,184,0.2);border-radius:0.45rem;">
        <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:${labelColor};">${label}</div>
        <div style="font-size:0.82rem;font-weight:800;color:${valueColor};">${value}</div>
      </div>
    `;

    grid.innerHTML = [
      cell("RMSE", rmse),
      cell("MAE", mae),
      cell("NASA", nasa),
      cell("Best Epoch", bestEpoch),
    ].join("");

    foot.textContent = `Dataset: ${dataset} | Device: ${String(device).toUpperCase()} | Updated: ${new Date().toLocaleTimeString()}`;
  }

  async function refreshMetricsDock(showFeedback = false) {
    try {
      const summary = await fetchSummary();
      setMetrics(summary);
      updateDockTheme(document.documentElement.dataset.theme || "dark");
      if (showFeedback) showToast("Model metrics refreshed.", "success");
    } catch (error) {
      setMetrics({
        metrics: { testRmse: null, testMae: null, nasaScore: null, bestEpoch: null },
        dataset: "offline",
        device: "unknown",
      });
      if (showFeedback) showToast("Metrics refresh failed. API may be offline.", "danger");
    }
  }

  function initMetricsDock() {
    const dock = ensureMetricsDock();
    const refreshBtn = dock.querySelector("#aeropredict-metrics-refresh");
    if (refreshBtn && refreshBtn.dataset.uiWired !== "true") {
      refreshBtn.dataset.uiWired = "true";
      refreshBtn.addEventListener("click", () => {
        refreshMetricsDock(true);
      });
    }

    refreshMetricsDock(false);
    restartMetricsInterval();
  }

  window.addEventListener("aeropredict:theme", (event) => {
    const theme = event?.detail?.theme || document.documentElement.dataset.theme || "dark";
    updateDockTheme(theme);
  });

  function wireHashLinks() {
    document.querySelectorAll('a[href="#"]').forEach((link) => {
      if (link.dataset.uiWired === "true") return;
      link.dataset.uiWired = "true";

      const label = normalizeText(link.textContent);
      link.addEventListener("click", async (event) => {
        event.preventDefault();

        if (label.includes("support")) {
          showToast("Support: Use TROUBLESHOOTING.md and dashboard API health checks.", "info");
          return;
        }
        if (label.includes("settings")) {
          openSettingsModal();
          return;
        }
        if (label.includes("security policy")) {
          showToast("Security: Use local host only and never expose raw telemetry publicly.", "warning");
          return;
        }
        if (label.includes("system health") || label.includes("system status")) {
          try {
            const summary = await fetchSummary();
            showToast(`System healthy on ${String(summary.device).toUpperCase()} (${summary.dataset}).`, "success");
          } catch (error) {
            showToast("System health check failed. Verify API server is running.", "danger");
          }
          return;
        }
        if (label.includes("documentation")) {
          window.open(`${getApiBase()}/api/summary`, "_blank", "noopener,noreferrer");
          return;
        }

        showToast("This control has no assigned action yet.", "info");
      });
    });
  }

  function wireButtonsAndIcons() {
    document.querySelectorAll("button, span").forEach((node) => {
      if (node.dataset.uiWired === "true") return;
      const icon = normalizeText(node.getAttribute("data-icon"));
      const text = normalizeText(node.textContent);

      const isClickableSpan = node.tagName === "SPAN" && (icon === "settings" || icon === "notifications");
      const isTargetButton = node.tagName === "BUTTON";
      if (!isClickableSpan && !isTargetButton) return;

      node.dataset.uiWired = "true";

      node.addEventListener("click", async () => {
        if (icon === "search") {
          const searchInput = node.parentElement ? node.parentElement.querySelector("input") : null;
          if (searchInput) {
            runSearchQuery(searchInput.value);
            searchInput.focus();
          }
          return;
        }

        if (icon === "notifications" || text === "notifications") {
          openNotificationsPanel();
          return;
        }
        if (icon === "settings" || text === "settings") {
          openSettingsModal();
          return;
        }

        if (text.includes("export config")) {
          try {
            const summary = await fetchSummary();
            downloadJsonFile("aeropredict-model-config.json", summary);
            showToast("Model config exported.", "success");
          } catch {
            showToast("Config export failed. API may be offline.", "danger");
          }
          return;
        }

        if (text.includes("deploy model")) {
          showToast("Deployment simulation started.", "warning");
          setTimeout(() => showToast("Deployment simulation complete.", "success"), 1200);
          return;
        }

        if (text.includes("run pipeline simulation")) {
          showToast("Pipeline simulation running...", "warning");
          setTimeout(() => showToast("Pipeline simulation completed.", "success"), 1400);
          return;
        }

        if (text.includes("sync stream")) {
          window.dispatchEvent(new CustomEvent("aeropredict:sync-stream"));
          showToast("Live stream sync triggered.", "success");
          return;
        }

        if (text.includes("last 24 hours") || text.includes("last 7 days") || text.includes("last 30 days")) {
          const labelNode = node.querySelector("span");
          const current = normalizeText(labelNode ? labelNode.textContent : node.textContent);
          const currentIndex = TIME_RANGES.findIndex((item) => normalizeText(item) === current);
          const nextLabel = TIME_RANGES[(Math.max(currentIndex, 0) + 1) % TIME_RANGES.length];
          if (labelNode) {
            labelNode.textContent = nextLabel;
          } else {
            node.textContent = nextLabel;
          }
          window.dispatchEvent(new CustomEvent("aeropredict:time-range", { detail: { range: nextLabel } }));
          showToast(`Time range set to ${nextLabel}.`, "info");
          return;
        }

        if (node.title && normalizeText(node.title).includes("filter columns")) {
          showToast("Column filter preset applied.", "info");
          return;
        }
        if (node.title && normalizeText(node.title).includes("grid settings")) {
          showToast("Grid settings opened (compact mode).", "info");
          return;
        }
      });
    });
  }

  function initUiFeatures() {
    ensureUiFeatureStyles();
    ensureSettingsModal();
    ensureNotificationsPanel();
    applyUiSettings();
    ensureNotificationBadges();
    refreshNotifications(false);
    startNotificationsPolling();
    wireSearchInputs();
    wireHashLinks();
    wireButtonsAndIcons();
    if (document.body && document.body.dataset.floatingMetrics === "true") {
      initMetricsDock();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initUiFeatures, { once: true });
  } else {
    initUiFeatures();
  }

  window.AeroPredictUi = {
    showToast,
    getApiBase,
    initUiFeatures,
    refreshMetricsDock,
    refreshNotifications,
    openSettingsModal,
    openNotificationsPanel,
    setUnreadNotifications,
    runSearchQuery,
  };
})();
