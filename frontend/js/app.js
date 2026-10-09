/**
 * Shared frontend behaviour, used by the landing page and the dashboard.
 *
 *   1. mobile navigation toggle
 *   2. a live health probe against the FastAPI backend
 *
 * The dashboard's own data loading lives in js/dashboard.js; this file only
 * covers the chrome the two pages share.
 */
(function () {
  'use strict';

  const config = window.TRAFFIC_CONFIG || {};

  /* ------------------------------------------------------------------ *
   * Mobile navigation
   * ------------------------------------------------------------------ */
  function initNavToggle() {
    const toggle = document.getElementById('nav-toggle');
    const nav = document.getElementById('primary-nav');
    if (!toggle || !nav) return;

    toggle.addEventListener('click', function () {
      const isOpen = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });
  }

  /* ------------------------------------------------------------------ *
   * API health probe
   * ------------------------------------------------------------------ */
  function setStatus(element, state, message) {
    element.dataset.state = state;
    element.textContent = message;
  }

  function showOutput(element, payload) {
    element.textContent = JSON.stringify(payload, null, 2);
    element.hidden = false;
  }

  async function checkApiHealth() {
    const statusEl = document.getElementById('api-status');
    const outputEl = document.getElementById('api-output');
    const retryBtn = document.getElementById('api-retry');
    if (!statusEl || !outputEl) return;

    if (!config.API_BASE_URL) {
      setStatus(statusEl, 'error', 'API_BASE_URL is not set in js/config.js.');
      return;
    }

    const healthUrl = config.API_BASE_URL + config.HEALTH_PATH;
    const urlEl = document.getElementById('api-health-url');
    if (urlEl) urlEl.textContent = healthUrl;

    if (retryBtn) retryBtn.disabled = true;
    setStatus(statusEl, 'pending', 'Checking API status…');

    const controller = new AbortController();
    const timer = window.setTimeout(
      function () { controller.abort(); },
      config.REQUEST_TIMEOUT_MS || 5000
    );

    try {
      const response = await fetch(healthUrl, { signal: controller.signal });
      const body = await response.json();
      showOutput(outputEl, body);

      if (response.ok) {
        setStatus(
          statusEl,
          'ok',
          'API is healthy — ' + (body.service || 'unknown service') +
            ' reported status "' + (body.status || 'unknown') + '".'
        );
      } else {
        setStatus(
          statusEl,
          'error',
          'API responded with HTTP ' + response.status + '.'
        );
      }
    } catch (error) {
      const reason = error.name === 'AbortError' ? 'request timed out' : error.message;
      outputEl.hidden = true;
      setStatus(
        statusEl,
        'error',
        'Could not reach the API at ' + healthUrl + ' (' + reason +
          '). Start it with: uvicorn app.main:app --reload'
      );
    } finally {
      window.clearTimeout(timer);
      if (retryBtn) retryBtn.disabled = false;
    }
  }

  function initApiCheck() {
    const retryBtn = document.getElementById('api-retry');
    if (retryBtn) {
      retryBtn.addEventListener('click', checkApiHealth);
    }
    checkApiHealth();
  }

  document.addEventListener('DOMContentLoaded', function () {
    initNavToggle();
    initApiCheck();
  });
})();