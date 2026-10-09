/**
 * Frontend runtime configuration.
 *
 * This file contains no secrets. It only tells the browser where to find the
 * local API and how the dashboard should behave. Change API_BASE_URL if the
 * backend runs on another host or port.
 *
 * Nothing here is a credential: the TomTom key, the database DSN and every
 * other environment variable live in the backend's .env and are never read by
 * the browser. The frontend talks to the API only.
 */
window.TRAFFIC_CONFIG = {
  /**
   * Where the API lives, without a trailing slash.
   *
   * An empty string means "the same origin the dashboard was loaded from",
   * which is correct for every single-URL deployment where the API serves the
   * dashboard itself: the Docker image, Render, or a local ``uvicorn`` run
   * (open http://127.0.0.1:8000/ or /dashboard.html).
   *
   * Set an absolute URL only when the dashboard and the API are served from
   * different origins, e.g. a static live-server preview on :5500 against the
   * API on :8000:
   *
   *   API_BASE_URL: 'http://127.0.0.1:8000'
   */
  API_BASE_URL: '',
  HEALTH_PATH: '/api/health',

  /** Per-request timeout. A hung backend must not hang the UI. */
  REQUEST_TIMEOUT_MS: 8000,

  /** Default automatic refresh. 0 disables it; the user can override in the UI. */
  REFRESH_INTERVAL_MS: 30000,

  /** Choices offered in the refresh selector. 0 means "off". */
  REFRESH_INTERVAL_OPTIONS: [15000, 30000, 60000, 0],

  /** Trend windows offered on the analytics chart, in hours. */
  TREND_HOUR_OPTIONS: [1, 6, 12, 24],
  DEFAULT_TREND_HOURS: 6,

  /** Rows requested from the prediction history endpoint. */
  PREDICTION_HISTORY_LIMIT: 20,

  /**
   * OpenStreetMap-compatible raster tiles. No key is required and none is
   * used. If these fail to load the map card reports it and every other panel
   * keeps working.
   */
  MAP_TILE_URL: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  MAP_TILE_ATTRIBUTION: '&copy; OpenStreetMap contributors',
  MAP_MIN_ZOOM: 3,
  MAP_MAX_ZOOM: 19
};
