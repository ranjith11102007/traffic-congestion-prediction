/**
 * Reusable API client for the dashboard.
 *
 * One place decides how a request is built, how long it may take, how a failure
 * is classified and what a human being is told about it. Every panel consumes
 * this module, so the dashboard reports backend trouble the same way
 * everywhere.
 *
 * Failure kinds:
 *   network  — the backend could not be reached at all
 *   timeout  — no answer within config.REQUEST_TIMEOUT_MS
 *   http     — the backend answered with a non-2xx status
 *   malformed— a 2xx body that is not the JSON the endpoint promises
 *
 * None of them ever surfaces a stack trace, a DSN or a vendor response body.
 */
(function () {
  'use strict';

  var config = window.TRAFFIC_CONFIG || {};

  /** An expected, explainable API failure. Carries no internals. */
  function ApiError(kind, message, options) {
    var opts = options || {};
    this.name = 'ApiError';
    this.kind = kind;
    this.message = message;
    this.status = opts.status || 0;
    this.code = opts.code || null;
    this.url = opts.url || '';
    this.retryable = opts.retryable !== false;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  /* ------------------------------------------------------------------ *
   * Human-readable messages
   * ------------------------------------------------------------------ */

  var HTTP_MESSAGES = {
    400: 'The request was rejected as invalid.',
    404: 'That resource does not exist yet.',
    409: 'The data is older than the freshness limit you asked for.',
    422: 'The API rejected the query parameters.',
    500: 'The backend hit an unexpected error. Check the server logs.',
    502: 'The backend returned an unusable response.',
    503: 'The service is temporarily unavailable.',
    504: 'The backend took too long to answer.'
  };

  function messageForStatus(status, code) {
    if (code === 'no_prediction_available') {
      return 'No prediction has been stored yet.';
    }
    if (code === 'database_unavailable') {
      return 'The database is unreachable right now.';
    }
    if (code === 'model_not_available') {
      return 'No trained model is loaded.';
    }
    if (code === 'insufficient_history') {
      return 'Not enough stored history to predict for that location yet.';
    }
    if (code === 'stale_data') {
      return 'The newest reading is older than the freshness limit.';
    }
    if (HTTP_MESSAGES[status]) return HTTP_MESSAGES[status];
    if (status >= 500) return 'The backend reported a server error (HTTP ' + status + ').';
    if (status >= 400) return 'The request could not be accepted (HTTP ' + status + ').';
    return 'Unexpected response from the API (HTTP ' + status + ').';
  }

  /* ------------------------------------------------------------------ *
   * Request plumbing
   * ------------------------------------------------------------------ */

  function buildUrl(path, params) {
    var base = String(config.API_BASE_URL || '').replace(/\/+$/, '');
    var url = base + path;
    if (params) {
      var pairs = [];
      Object.keys(params).forEach(function (key) {
        var value = params[key];
        if (value === undefined || value === null || value === '') return;
        pairs.push(encodeURIComponent(key) + '=' + encodeURIComponent(value));
      });
      if (pairs.length) url += '?' + pairs.join('&');
    }
    return url;
  }

  async function request(path, params) {
    if (!config.API_BASE_URL) {
      throw new ApiError(
        'configuration',
        'API_BASE_URL is not set in js/config.js.',
        { retryable: false }
      );
    }

    var url = buildUrl(path, params);
    var controller = new AbortController();
    var timer = window.setTimeout(function () {
      controller.abort();
    }, config.REQUEST_TIMEOUT_MS || 8000);

    var response;
    try {
      response = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: 'application/json' }
      });
    } catch (error) {
      window.clearTimeout(timer);
      if (error && error.name === 'AbortError') {
        throw new ApiError(
          'timeout',
          'The API did not answer within ' +
            Math.round((config.REQUEST_TIMEOUT_MS || 8000) / 1000) +
            ' seconds.',
          { url: url }
        );
      }
      throw new ApiError(
        'network',
        'Could not reach the API at ' + (config.API_BASE_URL || url) + '.',
        { url: url }
      );
    }
    window.clearTimeout(timer);

    var payload = null;
    var parseFailed = false;
    try {
      var text = await response.text();
      payload = text ? JSON.parse(text) : null;
    } catch (error) {
      parseFailed = true;
    }

    if (!response.ok) {
      var err = (payload && payload.error) || {};
      throw new ApiError('http', messageForStatus(response.status, err.code), {
        status: response.status,
        code: err.code || null,
        url: url,
        retryable: response.status >= 500 || response.status === 429
      });
    }

    if (parseFailed || (payload !== null && typeof payload !== 'object')) {
      throw new ApiError(
        'malformed',
        'The API returned a response the dashboard could not read.',
        { status: response.status, url: url }
      );
    }

    return payload;
  }

  /* ------------------------------------------------------------------ *
   * Endpoints the dashboard consumes
   * ------------------------------------------------------------------ */

  var api = {
    ApiError: ApiError,

    /** GET /api/health — liveness, used for the header status pill. */
    getHealth: function () {
      return request(config.HEALTH_PATH || '/api/health');
    },

    /** GET /api/dashboard/current — newest reading and forecast per location. */
    getCurrentDashboard: function () {
      return request('/api/dashboard/current');
    },

    /** GET /api/dashboard/locations — per-location readiness and coordinates. */
    getLocations: function () {
      return request('/api/dashboard/locations');
    },

    /**
     * GET /api/dashboard/trends — chart series.
     * @param {{location_id?: string, hours?: number, limit?: number}} params
     */
    getTrends: function (params) {
      return request('/api/dashboard/trends', params || {});
    },

    /** GET /api/dashboard/status — database, model, provider, scheduler. */
    getSystemStatus: function () {
      return request('/api/dashboard/status');
    },

    /** GET /api/dashboard/model — artifact provenance, metrics, limits. */
    getModelInfo: function () {
      return request('/api/dashboard/model');
    },

    /**
     * GET /api/traffic/predictions/latest — newest forecast, system-wide or
     * for one location. Throws ApiError('http', ..., code
     * 'no_prediction_available') when nothing has been predicted yet; callers
     * turn that into an empty state, not an error state.
     */
    getLatestPrediction: function (locationId) {
      return request('/api/traffic/predictions/latest', {
        location_id: locationId || undefined
      });
    },

    /**
     * GET /api/traffic/predictions — stored forecast history.
     * @param {{location_id?: string, limit?: number}} params
     */
    getPredictions: function (params) {
      return request('/api/traffic/predictions', params || {});
    }
  };

  window.TrafficAPI = api;
})();
