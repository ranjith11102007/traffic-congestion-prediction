/**
 * Shared display helpers: state blocks, formatting and labels.
 *
 * Three rules are enforced here so no panel has to remember them:
 *
 *   1. While loading, a panel says so in words. It never shows a number.
 *   2. A failure shows a friendly sentence and, where retrying makes sense, a
 *      retry button. Never a stack trace.
 *   3. Missing data renders as "N/A" or an explicit empty state. A zero is a
 *      measurement, so it is never used to mean "nothing here".
 */
(function () {
  'use strict';

  var NA = 'N/A';

  /* ------------------------------------------------------------------ *
   * Escaping — every API-derived string goes through this before it is
   * placed in innerHTML.
   * ------------------------------------------------------------------ */
  function esc(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /* ------------------------------------------------------------------ *
   * State blocks: loading / error / empty
   * ------------------------------------------------------------------ */

  function blockHtml(kind, title, detail, retryId) {
    var html =
      '<div class="state-block state-block--' + kind + '" role="status">' +
      '<p class="state-block__title">' + esc(title) + '</p>';
    if (detail) {
      html += '<p class="state-block__detail">' + esc(detail) + '</p>';
    }
    if (retryId) {
      html +=
        '<button type="button" class="button button--ghost state-block__retry" ' +
        'data-retry="' + esc(retryId) + '">Retry</button>';
    }
    return html + '</div>';
  }

  function setLoading(el, message) {
    if (!el) return;
    el.innerHTML = blockHtml('loading', message || 'Loading…');
  }

  function setError(el, message, detail, retryId) {
    if (!el) return;
    el.innerHTML = blockHtml('error', message, detail, retryId);
  }

  function setEmpty(el, title, detail) {
    if (!el) return;
    el.innerHTML = blockHtml('empty', title, detail);
  }

  /**
   * Render an ApiError into a container with a message that depends on how the
   * request failed, and offer a retry unless retrying is pointless.
   */
  function renderApiError(el, error, fallbackMessage, retryId) {
    if (!el) return;
    var message = fallbackMessage || 'Data is temporarily unavailable.';
    var detail = null;

    if (error && error.kind === 'network') {
      message = 'The backend is not reachable.';
      detail = 'Start it with: uvicorn app.main:app --reload';
    } else if (error && error.kind === 'timeout') {
      message = 'The request timed out.';
      detail = 'The backend is running slowly or is not responding.';
    } else if (error && error.kind === 'malformed') {
      message = 'The backend returned an unreadable response.';
      detail = 'This usually means a proxy is answering instead of the API.';
    } else if (error && error.message) {
      message = error.message;
    }

    setError(el, message, detail, retryId || null);
  }

  /* ------------------------------------------------------------------ *
   * Formatting
   * ------------------------------------------------------------------ */

  /** ISO string or Date -> local "HH:MM:SS"; null -> "N/A". */
  function formatTime(value) {
    if (!value) return NA;
    var date = value instanceof Date ? value : new Date(value);
    if (isNaN(date.getTime())) return NA;
    return date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  /** ISO string or Date -> local date and time; null -> "N/A". */
  function formatDateTime(value) {
    if (!value) return NA;
    var date = value instanceof Date ? value : new Date(value);
    if (isNaN(date.getTime())) return NA;
    return date.toLocaleString([], {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  /** Seconds -> "42s" / "12m" / "3h"; null -> "N/A". */
  function formatAge(seconds) {
    if (seconds === null || seconds === undefined || isNaN(seconds)) return NA;
    var s = Math.max(0, Math.round(Number(seconds)));
    if (s < 60) return s + 's';
    if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
    return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
  }

  /** Number of decimals, dropping trailing zeros; null -> "N/A". */
  function formatNumber(value, decimals) {
    if (value === null || value === undefined || value === '' || isNaN(value)) {
      return NA;
    }
    var digits = decimals === undefined ? 1 : decimals;
    return Number(value).toFixed(digits);
  }

  /** Percentage in 0..1 -> "86.4%", null -> "Confidence unavailable". */
  function formatConfidence(value) {
    if (value === null || value === undefined || isNaN(value)) {
      return 'Confidence unavailable';
    }
    return (Number(value) * 100).toFixed(1) + '%';
  }

  /** "free_flow" -> "Free flow". Unknown labels pass through prettified. */
  function titleCase(value) {
    if (value === null || value === undefined || value === '') return NA;
    return String(value)
      .split(/[_\s]+/)
      .filter(Boolean)
      .map(function (word) {
        return word.charAt(0).toUpperCase() + word.slice(1);
      })
      .join(' ');
  }

  /**
   * Congestion class id -> label, using the model's own names.
   * The frontend never invents a classification system of its own.
   */
  function congestionLabel(level, labelMap) {
    if (level === null || level === undefined) return null;
    var key = String(level);
    if (labelMap && labelMap[key]) return titleCase(labelMap[key]);
    var fallback = { 0: 'Free flow', 1: 'Moderate', 2: 'Heavy', 3: 'Severe' };
    return fallback[key] || null;
  }

  /** CSS modifier for a congestion label; safe because the map is fixed. */
  function congestionTone(level) {
    if (level === null || level === undefined) return 'unknown';
    var n = Number(level);
    if (n === 0) return 'low';
    if (n === 1) return 'moderate';
    if (n === 2) return 'high';
    if (n === 3) return 'severe';
    return 'unknown';
  }

  /** Freshness literal -> human wording. */
  function freshnessText(value) {
    if (value === 'fresh') return 'Fresh';
    if (value === 'stale') return 'Stale';
    if (value === 'unavailable') return 'No data';
    return NA;
  }

  /** Prediction availability literal -> human wording. */
  function predictionStatusText(value) {
    var map = {
      available: 'Available',
      insufficient_history: 'Waiting for history',
      unscoreable_location: 'Location not scoreable',
      model_unavailable: 'Model unavailable',
      unavailable: 'Not available'
    };
    return map[value] || NA;
  }

  /** Data source provenance, shown verbatim in meaning. */
  function sourceText(value, isSimulation) {
    if (isSimulation) return 'simulation';
    if (value) return value;
    return NA;
  }

  window.TrafficUI = {
    NA: NA,
    esc: esc,
    setLoading: setLoading,
    setError: setError,
    setEmpty: setEmpty,
    renderApiError: renderApiError,
    formatTime: formatTime,
    formatDateTime: formatDateTime,
    formatAge: formatAge,
    formatNumber: formatNumber,
    formatConfidence: formatConfidence,
    titleCase: titleCase,
    congestionLabel: congestionLabel,
    congestionTone: congestionTone,
    freshnessText: freshnessText,
    predictionStatusText: predictionStatusText,
    sourceText: sourceText
  };
})();
