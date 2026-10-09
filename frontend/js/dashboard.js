/**
 * Stage 7 dashboard controller.
 *
 * Responsibilities:
 *   - load the Stage 6 endpoints and paint each panel
 *   - keep every panel in one of four states: loading, loaded, empty, error
 *   - drive the map, the chart, the location selector and the auto-refresh
 *
 * Honesty rules this file exists to enforce:
 *   - a value is only ever printed if the API returned it; otherwise the panel
 *     says "N/A" or shows an explicit empty state;
 *   - an empty state explains what would make the data appear, so nobody has to
 *     guess why a panel is blank;
 *   - refreshes are on a single timer with a fixed interval, and one failing
 *     panel never stops the others from updating.
 */
(function () {
  'use strict';

  var config = window.TRAFFIC_CONFIG || {};
  var API = window.TrafficAPI;
  var UI = window.TrafficUI;

  var state = {
    current: null,
    locations: null,
    status: null,
    model: null,
    selectedId: null,
    hours: config.DEFAULT_TREND_HOURS || 6,
    intervalMs: config.REFRESH_INTERVAL_MS || 30000,
    timer: null,
    lastUpdated: null,
    running: false,
    connected: null
  };

  /* ------------------------------------------------------------------ *
   * Small helpers
   * ------------------------------------------------------------------ */

  function $(id) {
    return document.getElementById(id);
  }

  function setText(id, value, tone) {
    var el = $(id);
    if (!el) return;
    el.textContent = value;
    if (tone) el.setAttribute('data-tone', tone);
    else el.removeAttribute('data-tone');
  }

  function setConn(stateName, text) {
    var pill = $('header-conn');
    var label = $('header-conn-text');
    if (pill) pill.setAttribute('data-state', stateName);
    if (label) label.textContent = text;
  }

  function showBanner() {
    var banner = $('connection-banner');
    if (banner) banner.hidden = false;
  }

  function hideBanner() {
    var banner = $('connection-banner');
    if (banner) banner.hidden = true;
  }

  function markUpdated() {
    state.lastUpdated = new Date();
    var el = $('last-updated');
    if (el) {
      el.textContent = UI.formatTime(state.lastUpdated);
      el.setAttribute('datetime', state.lastUpdated.toISOString());
    }
  }

  function isOffline(error) {
    return !!error && (error.kind === 'network' || error.kind === 'timeout');
  }

  /** Record connectivity after a batch: any success clears the banner. */
  function reportConnectivity(offlineSeen, anySuccess) {
    if (anySuccess) {
      state.connected = true;
      hideBanner();
      if (state.status && state.status.status && state.status.status !== 'healthy') {
        setConn('degraded', 'API connected \u00B7 ' + state.status.status);
      } else {
        setConn('ok', 'API connected');
      }
      return;
    }
    if (offlineSeen) {
      state.connected = false;
      showBanner();
      setConn('error', 'API unreachable');
      return;
    }
    setConn('degraded', 'API responding with errors');
    showBanner();
  }

  function badge(text, tone) {
    return (
      '<span class="badge-status" data-tone="' +
      UI.esc(tone) +
      '">' +
      UI.esc(text) +
      '</span>'
    );
  }

  function sourceBadge(source, isSimulation) {
    var label = UI.sourceText(source, isSimulation);
    return (
      '<span class="badge-source" data-simulated="' +
      (isSimulation ? 'true' : 'false') +
      '">' +
      UI.esc(label) +
      '</span>'
    );
  }

  function congestionBadge(level, labelMap) {
    var label = UI.congestionLabel(level, labelMap);
    if (!label) return '<span class="badge-status" data-tone="unknown">No forecast</span>';
    return badge(label, UI.congestionTone(level));
  }

  function labelMap() {
    if (state.model && state.model.class_labels) return state.model.class_labels;
    return {};
  }

  /* ------------------------------------------------------------------ *
   * Selector
   * ------------------------------------------------------------------ */

  function populateSelector() {
    var select = $('location-select');
    if (!select || !state.locations) return;

    select.disabled = false;

    var previous = state.selectedId;
    var entries = state.locations.locations || [];
    var options = ['<option value="">All locations</option>'];

    entries.forEach(function (entry) {
      options.push(
        '<option value="' +
          UI.esc(entry.location_id) +
          '">' +
          UI.esc(entry.road_name || entry.location_id) +
          ' (' +
          UI.esc(entry.location_id) +
          ')</option>'
      );
    });

    select.innerHTML = options.join('');

    var stillExists =
      previous &&
      entries.some(function (entry) {
        return entry.location_id === previous;
      });

    state.selectedId = stillExists ? previous : null;
    select.value = state.selectedId || '';
  }

  function locationById(id) {
    if (!state.locations) return null;
    var found = null;
    (state.locations.locations || []).forEach(function (entry) {
      if (entry.location_id === id) found = entry;
    });
    return found;
  }

  function currentById(id) {
    if (!state.current) return null;
    var found = null;
    (state.current.locations || []).forEach(function (entry) {
      if (entry.location_id === id) found = entry;
    });
    return found;
  }

  /** Locations the table shows: the selected one, or all of them. */
  function visibleCurrentLocations() {
    if (!state.current) return [];
    if (!state.selectedId) return state.current.locations || [];
    var one = currentById(state.selectedId);
    return one ? [one] : [];
  }

  /* ------------------------------------------------------------------ *
   * KPI cards
   * ------------------------------------------------------------------ */

  function renderKpis() {
    var current = state.current;
    var status = state.status;

    // Monitored locations
    var monitored = null;
    if (status && typeof status.monitored_location_count === 'number') {
      monitored = status.monitored_location_count;
    } else if (current && typeof current.location_count === 'number') {
      monitored = current.location_count;
    }
    setText('kpi-locations', monitored === null ? UI.NA : String(monitored));
    if (current) {
      setText(
        'kpi-locations-sub',
        current.locations_with_data + ' of ' + current.location_count + ' have stored readings'
      );
    } else {
      setText('kpi-locations-sub', UI.NA);
    }

    // Locations with a forecast
    if (current && typeof current.predictions_available === 'number') {
      setText('kpi-predictions', String(current.predictions_available));
      setText(
        'kpi-predictions-sub',
        'of ' + current.location_count + ' monitored locations'
      );
    } else {
      setText('kpi-predictions', UI.NA);
      setText('kpi-predictions-sub', UI.NA);
    }

    // Latest mean speed, computed from returned readings only
    var speeds = [];
    if (current) {
      (current.locations || []).forEach(function (loc) {
        if (typeof loc.avg_speed_kph === 'number') speeds.push(loc.avg_speed_kph);
      });
    }
    if (speeds.length) {
      var mean = speeds.reduce(function (a, b) { return a + b; }, 0) / speeds.length;
      setText('kpi-speed', mean.toFixed(1) + ' km/h');
      setText(
        'kpi-speed-sub',
        'mean of ' + speeds.length + ' reporting location' + (speeds.length === 1 ? '' : 's')
      );
    } else {
      setText('kpi-speed', UI.NA);
      setText('kpi-speed-sub', 'No speed reported yet');
    }

    // Highest predicted level across locations with a stored forecast
    var levels = [];
    if (current) {
      (current.locations || []).forEach(function (loc) {
        if (loc.prediction && typeof loc.prediction.predicted_congestion_level === 'number') {
          levels.push(loc.prediction.predicted_congestion_level);
        }
      });
    }
    if (levels.length) {
      var worst = Math.max.apply(null, levels);
      var worstLabel = UI.congestionLabel(worst, labelMap()) || UI.NA;
      setText('kpi-congestion', worstLabel, UI.congestionTone(worst));
      setText(
        'kpi-congestion-sub',
        levels.length + ' location' + (levels.length === 1 ? '' : 's') + ' with a stored forecast'
      );
    } else {
      setText('kpi-congestion', UI.NA);
      setText('kpi-congestion-sub', 'No forecast stored yet');
    }

    // Data freshness
    var freshness = null;
    if (status && status.data_freshness) freshness = status.data_freshness;
    else if (current) freshness = current.locations && current.locations.length
      ? current.locations[0].freshness
      : null;
    if (freshness) {
      setText(
        'kpi-freshness',
        UI.freshnessText(freshness),
        freshness === 'fresh' ? 'low' : freshness === 'stale' ? 'moderate' : 'unknown'
      );
      var threshold = (status && status.freshness_threshold_seconds) ||
        (current && current.freshness_threshold_seconds);
      setText(
        'kpi-freshness-sub',
        threshold ? 'threshold ' + UI.formatAge(threshold) : UI.NA
      );
    } else {
      setText('kpi-freshness', UI.NA);
      setText('kpi-freshness-sub', UI.NA);
    }

    // Model version
    var version = (status && status.model_version) ||
      (state.model && state.model.model_version);
    setText('kpi-model', version || UI.NA);
    if (status && status.model_status === 'not_available') {
      setText('kpi-model-sub', 'No artifact loaded');
    } else if (state.model && state.model.dataset_is_simulated) {
      setText('kpi-model-sub', 'Trained on simulated development data');
    } else {
      setText('kpi-model-sub', version ? 'Loaded' : UI.NA);
    }
  }

  /* ------------------------------------------------------------------ *
   * Live traffic table
   * ------------------------------------------------------------------ */

  function renderCurrentTable() {
    var host = $('current-table');
    if (!host) return;

    var rows = visibleCurrentLocations();
    var withReadings = rows.filter(function (loc) {
      return loc.avg_speed_kph !== null;
    });

    if (!rows.length) {
      UI.setEmpty(
        host,
        'No locations to show',
        'The backend returned no monitored locations. Check GET /api/dashboard/locations.'
      );
      setText('current-count', '0 locations');
      return;
    }

    if (!withReadings.length) {
      UI.setEmpty(
        host,
        'No traffic observations yet',
        'Nothing has been collected for ' +
          (state.selectedId ? state.selectedId : 'any location') +
          '. Start the collector with COLLECTION_ENABLED=true or POST /api/traffic/collect.'
      );
      setText('current-count', '0 of ' + rows.length + ' locations have readings');
      return;
    }

    var head = [
      '<thead><tr>',
      '<th scope="col">Location</th>',
      '<th scope="col">Road</th>',
      '<th scope="col">Avg speed</th>',
      '<th scope="col">Free flow</th>',
      '<th scope="col">Congestion</th>',
      '<th scope="col">Temperature</th>',
      '<th scope="col">Weather</th>',
      '<th scope="col">Precipitation</th>',
      '<th scope="col">Data source</th>',
      '<th scope="col">Updated</th>',
      '<th scope="col">Freshness</th>',
      '</tr></thead>'
    ].join('');

    var body = rows
      .map(function (loc) {
        var selected = loc.location_id === state.selectedId;
        var level = loc.prediction
          ? loc.prediction.predicted_congestion_level
          : null;

        return (
          '<tr data-clickable="true" data-location="' +
          UI.esc(loc.location_id) +
          '" aria-selected="' +
          (selected ? 'true' : 'false') +
          '">' +
          '<td class="num">' + UI.esc(loc.location_id) + '</td>' +
          '<td>' + UI.esc(loc.road_name) + '</td>' +
          '<td class="num">' +
          (loc.avg_speed_kph === null
            ? '<span class="cell-muted">N/A</span>'
            : UI.formatNumber(loc.avg_speed_kph) + ' km/h') +
          '</td>' +
          '<td class="num">' +
          (loc.free_flow_speed_kph === null
            ? '<span class="cell-muted">N/A</span>'
            : UI.formatNumber(loc.free_flow_speed_kph) + ' km/h') +
          '</td>' +
          '<td>' +
          (loc.prediction
            ? congestionBadge(level, labelMap())
            : '<span class="badge-status" data-tone="unknown">' +
              UI.esc(UI.predictionStatusText(loc.prediction_status)) +
              '</span>') +
          '</td>' +
          '<td class="num">' +
          (loc.temperature_c === null
            ? '<span class="cell-muted">N/A</span>'
            : UI.formatNumber(loc.temperature_c) + ' \u00B0C') +
          '</td>' +
          '<td>' +
          (loc.weather_condition
            ? UI.esc(UI.titleCase(loc.weather_condition))
            : '<span class="cell-muted">N/A</span>') +
          '</td>' +
          '<td class="num">' +
          (loc.rainfall_mm === null
            ? '<span class="cell-muted">N/A</span>'
            : UI.formatNumber(loc.rainfall_mm) + ' mm') +
          '</td>' +
          '<td>' + sourceBadge(loc.data_source, loc.is_simulation) + '</td>' +
          '<td class="num">' +
          (loc.observation_timestamp
            ? UI.esc(UI.formatDateTime(loc.observation_timestamp))
            : '<span class="cell-muted">N/A</span>') +
          '</td>' +
          '<td>' +
          (loc.freshness === 'fresh'
            ? badge('Fresh', 'low')
            : loc.freshness === 'stale'
              ? badge('Stale', 'moderate')
              : badge('No data', 'unknown')) +
          '</td>' +
          '</tr>'
        );
      })
      .join('');

    host.innerHTML =
      '<div class="table-scroll"><table class="data-table">' +
      head +
      '<tbody>' +
      body +
      '</tbody></table></div>';

    setText(
      'current-count',
      withReadings.length +
        ' of ' +
        rows.length +
        ' location' +
        (rows.length === 1 ? '' : 's') +
        ' reporting'
    );
  }

  /* ------------------------------------------------------------------ *
   * Selected-location detail (traffic, weather, forecast)
   * ------------------------------------------------------------------ */

  function detailItem(label, value, note, tone) {
    return (
      '<div class="detail-item">' +
      '<span class="detail-item__label">' + UI.esc(label) + '</span>' +
      '<span class="detail-item__value"' +
      (tone ? ' data-tone="' + UI.esc(tone) + '"' : '') +
      '>' + UI.esc(value) + '</span>' +
      (note ? '<span class="detail-item__note">' + UI.esc(note) + '</span>' : '') +
      '</div>'
    );
  }

  function renderDetail() {
    var host = $('location-detail');
    if (!host) return;

    var roadLabel = $('detail-road');
    if (roadLabel) roadLabel.textContent = state.selectedId || 'All locations';

    if (!state.current) return; // loader already painted a loading state

    if (!state.selectedId) {
      var withData = (state.current.locations || []).filter(function (loc) {
        return loc.avg_speed_kph !== null;
      });
      UI.setEmpty(
        host,
        'No location selected',
        withData.length
          ? 'Pick a location in the sidebar, a map marker or a table row to see its reading, weather and forecast.'
          : 'Pick a location to inspect it. No location has stored readings yet.'
      );
      return;
    }

    var loc = currentById(state.selectedId);
    var meta = locationById(state.selectedId);

    if (!loc) {
      UI.setEmpty(
        host,
        'No data for ' + state.selectedId,
        'The backend returned no current reading for this location.'
      );
      return;
    }

    if (loc.avg_speed_kph === null && !loc.prediction) {
      UI.setEmpty(
        host,
        'No reading stored for this location',
        'This location has never been collected. POST /api/traffic/collect with this location id to store the first observation.'
      );
      return;
    }

    var html = [];

    html.push('<p class="detail-block-title">Traffic</p>');
    html.push('<div class="detail-grid">');
    html.push(detailItem('Average speed',
      loc.avg_speed_kph === null ? UI.NA : UI.formatNumber(loc.avg_speed_kph) + ' km/h',
      loc.observation_timestamp ? 'observed ' + UI.formatDateTime(loc.observation_timestamp) : null));
    html.push(detailItem('Free-flow speed',
      loc.free_flow_speed_kph === null ? UI.NA : UI.formatNumber(loc.free_flow_speed_kph) + ' km/h',
      null));
    html.push(detailItem('Freshness',
      UI.freshnessText(loc.freshness),
      loc.observation_age_seconds !== null
        ? 'age ' + UI.formatAge(loc.observation_age_seconds)
        : null,
      loc.freshness === 'fresh' ? 'low' : loc.freshness === 'stale' ? 'moderate' : 'unknown'));
    html.push(detailItem('Data source', UI.sourceText(loc.data_source, loc.is_simulation),
      loc.is_simulation ? 'synthetic development data' : null));
    html.push('</div>');

    html.push('<p class="detail-block-title">Weather</p>');
    html.push('<div class="detail-grid">');
    html.push(detailItem('Condition',
      loc.weather_condition ? UI.titleCase(loc.weather_condition) : UI.NA,
      state.status ? 'source: ' + state.status.weather_provider : null));
    html.push(detailItem('Temperature',
      loc.temperature_c === null ? UI.NA : UI.formatNumber(loc.temperature_c) + ' \u00B0C', null));
    html.push(detailItem('Precipitation',
      loc.rainfall_mm === null ? UI.NA : UI.formatNumber(loc.rainfall_mm) + ' mm', null));
    html.push('</div>');

    html.push('<p class="detail-block-title">Forecast</p>');
    if (loc.prediction) {
      var p = loc.prediction;
      html.push('<div class="detail-grid">');
      html.push(detailItem('Predicted congestion',
        UI.titleCase(p.predicted_congestion),
        'model ' + UI.titleCase(p.model_version),
        UI.congestionTone(p.predicted_congestion_level)));
      html.push(detailItem('Confidence',
        p.confidence === null || p.confidence === undefined
          ? 'Confidence unavailable'
          : UI.formatConfidence(p.confidence),
        p.confidence === null || p.confidence === undefined
          ? 'this estimator reports no probability'
          : 'highest class probability'));
      html.push(detailItem('Forecast made', UI.formatDateTime(p.prediction_timestamp), null));
      html.push(detailItem('Describes reading', UI.formatDateTime(p.observation_timestamp), null));
      html.push('</div>');
    } else {
      var reason = UI.predictionStatusText(loc.prediction_status);
      var help =
        loc.prediction_status === 'insufficient_history'
          ? 'Collecting historical observations. Prediction will become available after sufficient history is accumulated.'
          : loc.prediction_status === 'model_unavailable'
            ? 'No trained model artifact is loaded. Training one is required before forecasts can be produced.'
            : loc.prediction_status === 'unscoreable_location'
              ? 'This location is not mapped to a road segment the model was trained on.'
              : 'No prediction has been stored for this location yet.';
      UI.setEmpty(host, reason, help);
      return;
    }

    if (meta && meta.road_segment_id) {
      html.push(
        '<p class="panel__note" style="margin-top:0.8rem">Road segment ' +
          UI.esc(meta.road_segment_id) +
          '</p>'
      );
    }

    host.innerHTML = html.join('');
  }

  /* ------------------------------------------------------------------ *
   * Predictions
   * ------------------------------------------------------------------ */

  function renderLatestPrediction(prediction) {
    var host = $('latest-prediction');
    if (!host) return;

    $('latest-pred-scope').textContent = state.selectedId || 'All locations';

    if (!prediction) {
      return; // a loader already painted loading/error/empty
    }

    var tone = UI.congestionTone(prediction.predicted_congestion_level);
    var html = [];

    html.push('<div class="pred-headline">');
    html.push(
      '<span class="pred-headline__level" data-tone="' + UI.esc(tone) + '">' +
        UI.esc(UI.titleCase(prediction.predicted_congestion)) +
        '</span>'
    );
    html.push(
      '<span class="pred-headline__meta">model ' +
        UI.esc(prediction.model_version) +
        '</span>'
    );
    html.push('</div>');

    html.push('<div class="pred-grid" style="margin-top:0.9rem">');
    html.push(detailItem(
      'Confidence',
      UI.formatConfidence(prediction.confidence),
      prediction.confidence === null || prediction.confidence === undefined
        ? 'the estimator reported no probability'
        : 'highest class probability'
    ));
    html.push(detailItem('Predicted at', UI.formatDateTime(prediction.prediction_timestamp), null));
    html.push(detailItem('Describes reading', UI.formatDateTime(prediction.observation_timestamp), null));
    html.push(detailItem('Observation reference', String(prediction.observation_id), null));
    html.push(detailItem('Location', prediction.location_id, prediction.road_name));
    html.push(detailItem('Data source', UI.sourceText(prediction.source, prediction.source === 'simulation'),
      prediction.source === 'simulation' ? 'synthetic development data' : null));
    html.push('</div>');

    host.innerHTML = html.join('');
  }

  function renderPredictionHistory(page) {
    var host = $('prediction-history');
    if (!host) return;
    if (!page) return;

    var items = page.predictions || [];
    setText('history-count', page.count + ' of up to ' + page.limit + ' shown');

    if (!items.length) {
      UI.setEmpty(
        host,
        'No prediction available yet.',
        'Collecting historical observations. Prediction will become available after sufficient history is accumulated (the model needs six prior readings per road segment).'
      );
      return;
    }

    var head =
      '<thead><tr>' +
      '<th scope="col">Forecast time</th>' +
      '<th scope="col">Road</th>' +
      '<th scope="col">Location</th>' +
      '<th scope="col">Predicted</th>' +
      '<th scope="col">Confidence</th>' +
      '<th scope="col">Model</th>' +
      '<th scope="col">Source</th>' +
      '</tr></thead>';

    var body = items
      .map(function (item) {
        return (
          '<tr data-clickable="true" data-location="' + UI.esc(item.location_id) + '">' +
          '<td class="num">' + UI.esc(UI.formatDateTime(item.prediction_timestamp)) + '</td>' +
          '<td>' + UI.esc(item.road_name) + '</td>' +
          '<td class="num">' + UI.esc(item.location_id) + '</td>' +
          '<td>' + congestionBadge(item.predicted_congestion_level, labelMap()) + '</td>' +
          '<td class="num">' +
          (item.confidence === null || item.confidence === undefined
            ? '<span class="cell-muted">Confidence unavailable</span>'
            : UI.esc(UI.formatConfidence(item.confidence))) +
          '</td>' +
          '<td class="num">' + UI.esc(item.model_version) + '</td>' +
          '<td>' + sourceBadge(item.source, item.source === 'simulation') + '</td>' +
          '</tr>'
        );
      })
      .join('');

    host.innerHTML =
      '<div class="table-scroll"><table class="data-table">' +
      head +
      '<tbody>' +
      body +
      '</tbody></table></div>';
  }

  /* ------------------------------------------------------------------ *
   * Locations panel
   * ------------------------------------------------------------------ */

  function renderLocationsPanel() {
    var host = $('locations-grid');
    if (!host) return;
    if (!state.locations) return;

    var entries = state.locations.locations || [];

    if (!entries.length) {
      UI.setEmpty(
        host,
        'No locations configured',
        'config/monitored_locations.json did not yield any roads, so there is nothing to monitor.'
      );
      return;
    }

    var cards = entries
      .map(function (entry) {
        var required = entry.required_readings;
        var available = entry.available_readings || 0;
        var pct =
          required && required > 0
            ? Math.min(100, Math.round((available / required) * 100))
            : entry.observation_count > 0
              ? 100
              : 0;
        var complete = required !== null && required !== undefined && available >= required;

        var statusTone =
          entry.prediction_status === 'available'
            ? 'low'
            : entry.prediction_status === 'insufficient_history'
              ? 'moderate'
              : entry.prediction_status === 'model_unavailable'
                ? 'bad'
                : 'unknown';

        return (
          '<button type="button" class="location-card" data-location="' +
          UI.esc(entry.location_id) +
          '" aria-pressed="' +
          (entry.location_id === state.selectedId ? 'true' : 'false') +
          '">' +
          '<span class="location-card__top">' +
          '<span class="location-card__road">' + UI.esc(entry.road_name) + '</span>' +
          '<span class="location-card__id">' + UI.esc(entry.location_id) + '</span>' +
          '</span>' +
          '<span class="location-card__row">' +
          badge(UI.predictionStatusText(entry.prediction_status), statusTone) +
          '<span>' + UI.esc(entry.data_status) + '</span>' +
          '</span>' +
          '<span class="location-card__row">' +
          '<span>' +
          (required === null || required === undefined
            ? entry.observation_count + ' reading' + (entry.observation_count === 1 ? '' : 's')
            : available + ' of ' + required + ' readings') +
          '</span>' +
          '<span>' + entry.observation_count + ' stored</span>' +
          '</span>' +
          '<span class="reading-progress" role="img" aria-label="' +
          UI.esc(
            (required === null || required === undefined
              ? entry.observation_count + ' readings stored'
              : available + ' of ' + required + ' readings available') +
              ' for ' +
              entry.road_name
          ) +
          '">' +
          '<span class="reading-progress__fill" style="width:' +
          pct +
          '%" data-complete="' +
          (complete ? 'true' : 'false') +
          '"></span>' +
          '</span>' +
          '<span class="location-card__row">' +
          '<span>Observations: ' + entry.observation_count + '</span>' +
          '<span>' +
          (entry.latest_prediction_time
            ? 'forecast ' + UI.esc(UI.formatDateTime(entry.latest_prediction_time))
            : 'no forecast') +
          '</span>' +
          '</span>' +
          '<span class="location-card__row">' +
          sourceBadge(entry.source, entry.is_simulation) +
          '<span>' +
          (entry.freshness === 'fresh'
            ? 'Fresh'
            : entry.freshness === 'stale'
              ? 'Stale'
              : 'No data') +
          '</span>' +
          '</span>' +
          '</button>'
        );
      })
      .join('');

    host.innerHTML = '<div class="location-grid">' + cards + '</div>';
  }

  /* ------------------------------------------------------------------ *
   * System status panel
   * ------------------------------------------------------------------ */

  function renderStatusPanel() {
    var host = $('system-status');
    if (!host) return;
    if (!state.status) return;

    var s = state.status;

    var overallTone =
      s.status === 'healthy' ? 'low' : s.status === 'degraded' ? 'moderate' : 'severe';

    var rows = [
      ['API', badge(UI.titleCase(s.status), overallTone) + ' ' + UI.esc(s.service) + ' \u00B7 ' + UI.esc(s.environment)],
      [
        'Database',
        s.database_status === 'connected'
          ? badge('Connected', 'low')
          : s.database_status === 'not_configured'
            ? badge('Not configured', 'moderate')
            : badge('Unavailable', 'severe')
      ],
      [
        'Traffic provider',
        UI.esc(s.traffic_provider) +
          ' ' +
          (s.traffic_provider_configured
            ? badge('Configured', 'low')
            : badge('Not configured', 'moderate')) +
          (s.traffic_provider_is_simulation ? ' ' + badge('Simulated', 'moderate') : '')
      ],
      [
        'Weather provider',
        UI.esc(s.weather_provider) +
          ' ' +
          (s.weather_provider_configured
            ? badge('Enabled', 'low')
            : badge('Disabled', 'moderate'))
      ],
      [
        'Scheduler',
        s.scheduler_enabled
          ? badge(s.scheduler_running ? 'Running' : 'Enabled', 'low') +
            (s.scheduler_interval_seconds
              ? ' every ' + UI.esc(UI.formatAge(s.scheduler_interval_seconds))
              : '')
          : badge('Disabled', 'moderate') +
            (s.collection_cycles_completed || s.collection_cycles_failed
              ? ''
              : ' \u00B7 no background collection')
      ],
      [
        'ML model',
        s.model_status === 'available'
          ? badge('Loaded', 'low') + ' ' + UI.esc(s.model_version || UI.NA) +
            (s.model_trained_on_simulated_data ? ' ' + badge('Simulated data', 'moderate') : '')
          : badge('Not available', 'severe')
      ],
      [
        'Last successful collection',
        s.last_successful_collection_time
          ? UI.esc(UI.formatDateTime(s.last_successful_collection_time))
          : badge('Never', 'moderate')
      ],
      ['Monitored locations', UI.esc(String(s.monitored_location_count))],
      ['Stored observations', UI.esc(String(s.observation_count))],
      ['Stored forecasts', UI.esc(String(s.prediction_count))],
      [
        'Data freshness',
        s.data_freshness === 'fresh'
          ? badge('Fresh', 'low')
          : s.data_freshness === 'stale'
            ? badge('Stale', 'moderate')
            : badge('No data', 'unknown')
      ],
      [
        'Collection cycles',
        'completed ' +
          UI.esc(String(s.collection_cycles_completed)) +
          ' \u00B7 failed ' +
          UI.esc(String(s.collection_cycles_failed))
      ]
    ];

    var html = '<dl class="status-list">';
    rows.forEach(function (row) {
      html +=
        '<div class="status-row"><dt>' + UI.esc(row[0]) + '</dt><dd>' + row[1] + '</dd></div>';
    });
    html += '</dl>';

    if (s.last_error) {
      html +=
        '<div class="state-block state-block--error" style="margin-top:1rem">' +
        '<p class="state-block__title">Last collection error</p>' +
        '<p class="state-block__detail">' + UI.esc(s.last_error) + '</p>' +
        '</div>';
    }

    if (s.model_trained_on_simulated_data) {
      html +=
        '<div class="model-banner" style="margin-top:1rem">' +
        '<div><strong>Training data: simulated/development dataset.</strong> ' +
        'Real-world predictive performance has not yet been validated.</div>' +
        '</div>';
    }

    host.innerHTML = html;
  }

  /* ------------------------------------------------------------------ *
   * Model information panel
   * ------------------------------------------------------------------ */

  function renderModelPanel(model) {
    var host = $('model-info');
    if (!host) return;
    if (!model) return;

    setText('model-version', 'version ' + model.model_version);

    var html = [];

    if (model.dataset_is_simulated) {
      html.push(
        '<div class="model-banner">' +
          '<div><strong>Training data: simulated/development dataset.</strong> ' +
          'Real-world predictive performance has not yet been validated.</div>' +
          '</div>'
      );
    } else if (model.validated_on_real_traffic) {
      html.push(
        '<div class="model-banner model-banner--validated">' +
          '<div><strong>Training data: measured traffic data.</strong></div>' +
          '</div>'
      );
    }

    html.push('<div class="pred-grid">');
    html.push(detailItem('Model version', model.model_version, null));
    html.push(detailItem('Model type', model.model_display_name || model.model_type, model.model_type));
    html.push(detailItem('Trained at', UI.formatDateTime(model.trained_at), null));
    html.push(detailItem('Target', UI.titleCase(model.target_name), 'derived, never an input'));
    html.push(detailItem('Features', String(model.feature_count), 'model inputs'));
    html.push(detailItem('History required', model.min_history_per_segment + ' readings', 'per road segment'));
    html.push(detailItem('Split', model.split_strategy, 'hold-out ' + Math.round(model.split_ratio * 100) + '%'));
    html.push(detailItem('Dataset rows', String(model.dataset_rows_total), model.dataset_name));
    html.push('</div>');

    var metricKeys = Object.keys(model.metrics || {});
    if (metricKeys.length) {
      html.push('<p class="sub-head">Held-out evaluation</p>');
      html.push('<div class="pred-grid">');
      metricKeys.forEach(function (key) {
        var value = model.metrics[key];
        html.push(
          detailItem(
            UI.titleCase(key),
            (value * 100).toFixed(1) + '%',
            key === model.primary_metric ? 'primary metric' : null
          )
        );
      });
      if (typeof model.majority_class_baseline_accuracy === 'number') {
        html.push(
          detailItem(
            'Majority baseline',
            (model.majority_class_baseline_accuracy * 100).toFixed(1) + '%',
            'always predicting the common class'
          )
        );
      }
      html.push('</div>');
    }

    var labels = model.class_labels || {};
    var labelKeys = Object.keys(labels);
    if (labelKeys.length) {
      html.push('<p class="sub-head">Congestion classes</p>');
      html.push('<ul class="tag-list">');
      labelKeys.forEach(function (key) {
        html.push(
          '<li>' + UI.esc(key) + ' \u2192 ' + UI.esc(UI.titleCase(labels[key])) + '</li>'
        );
      });
      html.push('</ul>');
    }

    if (model.known_limitations && model.known_limitations.length) {
      html.push('<p class="sub-head">Known limitations</p>');
      html.push('<ul class="bullet-list">');
      model.known_limitations.forEach(function (limitation) {
        html.push('<li>' + UI.esc(limitation) + '</li>');
      });
      html.push('</ul>');
    }

    if (model.features && model.features.length) {
      html.push('<p class="sub-head">Inputs (' + model.features.length + ')</p>');
      html.push('<ul class="tag-list">');
      model.features.forEach(function (feature) {
        html.push('<li>' + UI.esc(feature) + '</li>');
      });
      html.push('</ul>');
    }

    host.innerHTML = html.join('');
  }

  /* ------------------------------------------------------------------ *
   * Loaders
   * ------------------------------------------------------------------ */

  async function loadCurrent() {
    var host = $('current-table');
    try {
      state.current = await API.getCurrentDashboard();
      renderKpis();
      renderCurrentTable();
      renderDetail();
      updateMap();
      return state.current;
    } catch (error) {
      state.current = null;
      renderKpis();
      UI.renderApiError(host, error, 'Traffic data is temporarily unavailable.', 'current');
      var detail = $('location-detail');
      UI.renderApiError(detail, error, 'Traffic data is temporarily unavailable.', 'current');
      throw error;
    }
  }

  async function loadLocations() {
    var host = $('locations-grid');
    try {
      state.locations = await API.getLocations();
      populateSelector();
      renderLocationsPanel();
      updateMap();
      var select = $('location-select');
      if (select && state.locations) {
        var hint = $('location-select-hint');
        if (hint) {
          hint.textContent =
            state.locations.location_count +
            ' location' +
            (state.locations.location_count === 1 ? '' : 's') +
            ' known to the backend.';
        }
      }
      return state.locations;
    } catch (error) {
      state.locations = null;
      UI.renderApiError(host, error, 'Locations are temporarily unavailable.', 'locations');
      var mapState = $('map-state');
      if (mapState) {
        mapState.hidden = false;
        mapState.innerHTML =
          '<span>Locations are temporarily unavailable. Use Retry to ask the API again.</span>';
      }
      var select = $('location-select');
      if (select) {
        select.innerHTML = '<option value="">Locations unavailable</option>';
        select.disabled = true;
      }
      throw error;
    }
  }

  async function loadStatus() {
    var host = $('system-status');
    try {
      state.status = await API.getSystemStatus();
      renderStatusPanel();
      renderKpis();
      return state.status;
    } catch (error) {
      state.status = null;
      UI.renderApiError(host, error, 'System status is temporarily unavailable.', 'status');
      renderKpis();
      throw error;
    }
  }

  async function loadModel() {
    var host = $('model-info');
    try {
      state.model = await API.getModelInfo();
      renderModelPanel(state.model);
      renderKpis();
      return state.model;
    } catch (error) {
      state.model = null;
      if (error && error.status === 503) {
        UI.setEmpty(
          host,
          'No trained model is loaded',
          'Train one with python -m ml.train, then reload. Until then no forecast can be produced.'
        );
      } else {
        UI.renderApiError(host, error, 'Model information is temporarily unavailable.', 'model');
      }
      setText('model-version', UI.NA);
      renderKpis();
      throw error;
    }
  }

  async function loadPrediction() {
    var host = $('latest-prediction');
    try {
      var prediction = await API.getLatestPrediction(state.selectedId);
      renderLatestPrediction(prediction);
      return prediction;
    } catch (error) {
      if (error && error.code === 'no_prediction_available') {
        UI.setEmpty(
          host,
          'No prediction available yet.',
          'Collecting historical observations. Prediction will become available after sufficient history is accumulated. The model needs ' +
            ((state.model && state.model.min_history_per_segment) || 6) +
            ' prior readings on a road segment before it will score.'
        );
        return null;
      }
      if (error && error.status === 404) {
        UI.setEmpty(
          host,
          'Unknown location',
          'The backend does not know "' + (state.selectedId || '') + '".'
        );
        return null;
      }
      UI.renderApiError(host, error, 'Predictions are temporarily unavailable.', 'prediction');
      throw error;
    }
  }

  async function loadHistory() {
    var host = $('prediction-history');
    try {
      var page = await API.getPredictions({
        location_id: state.selectedId || undefined,
        limit: config.PREDICTION_HISTORY_LIMIT || 20
      });
      renderPredictionHistory(page);
      return page;
    } catch (error) {
      UI.renderApiError(host, error, 'Forecast history is temporarily unavailable.', 'history');
      throw error;
    }
  }

  async function loadTrends() {
    var host = $('trend-loading');
    var location = state.selectedId;
    try {
      var trends = await API.getTrends({
        location_id: location || undefined,
        hours: state.hours
      });

      if (host) host.hidden = true;
      window.TrafficCharts.render(trends, { labelMap: labelMap() });

      var label = $('trend-location');
      if (label) label.textContent = location || 'all locations';

      var meta = $('trend-meta');
      if (meta) {
        meta.textContent =
          trends.point_count +
          ' point' +
          (trends.point_count === 1 ? '' : 's') +
          ' over ' +
          trends.hours +
          'h' +
          (trends.truncated ? ' (most recent slice of the window)' : '') +
          ' \u00B7 ' +
          UI.freshnessText(trends.freshness);
      }
      return trends;
    } catch (error) {
      if (host) {
        host.hidden = false;
        host.className = 'panel-loading';
        host.textContent = 'Trend data is temporarily unavailable.';
      }
      window.TrafficCharts.destroy();
      if (error && error.status === 404) {
        var notice = $('trend-notice');
        if (notice) {
          notice.hidden = false;
          notice.innerHTML =
            '<strong>Unknown location</strong><span>The backend does not know this location id.</span>';
        }
      }
      throw error;
    }
  }

  /* ------------------------------------------------------------------ *
   * Map
   * ------------------------------------------------------------------ */

  function updateMap() {
    if (!state.locations) return;
    var mapState = $('map-state');
    if (typeof window.L === 'undefined') {
      if (mapState) {
        mapState.hidden = false;
        mapState.innerHTML =
          '<span>Map library failed to load. Leaflet is served from frontend/vendor; every other panel still works.</span>';
      }
      return;
    }
    window.TrafficMap.update(state.locations.locations || [], state.current ? state.current.locations || [] : []);
    if (mapState) mapState.hidden = true;
  }

  /* ------------------------------------------------------------------ *
   * Selection
   * ------------------------------------------------------------------ */

  function selectLocation(id) {
    state.selectedId = id || null;

    var select = $('location-select');
    if (select && !select.disabled) select.value = state.selectedId || '';

    renderCurrentTable();
    renderDetail();

    document.querySelectorAll('.location-card').forEach(function (card) {
      card.setAttribute(
        'aria-pressed',
        card.getAttribute('data-location') === state.selectedId ? 'true' : 'false'
      );
    });

    window.TrafficMap.select(state.selectedId);

    // Forecast, history and chart all depend on the selection. Each loader
    // paints its own error state, so the rejection is already reported.
    resetPredictionPanels();
    loadPrediction().catch(function () {});
    loadHistory().catch(function () {});
    loadTrends().catch(function () {});
  }

  function resetPredictionPanels() {
    var latest = $('latest-prediction');
    var history = $('prediction-history');
    UI.setLoading(latest, 'Loading predictions…');
    UI.setLoading(history, 'Loading predictions…');
    var loading = $('trend-loading');
    if (loading) {
      loading.hidden = false;
      loading.className = 'panel-loading';
      loading.textContent = 'Loading analytics…';
    }
    $('latest-pred-scope').textContent = state.selectedId || 'All locations';
    $('trend-location').textContent = state.selectedId || 'all locations';
  }

  /* ------------------------------------------------------------------ *
   * Refresh cycle
   * ------------------------------------------------------------------ */

  async function refreshAll(options) {
    var opts = options || {};
    if (state.running) return;
    state.running = true;

    var button = $('refresh-btn');
    if (button) {
      button.disabled = true;
      button.setAttribute('data-busy', 'true');
      button.textContent = 'Refreshing…';
    }

    var offlineSeen = false;
    var anySuccess = false;

    var tasks = [
      loadCurrent(),
      loadLocations(),
      loadStatus(),
      loadPrediction(),
      loadHistory(),
      loadTrends()
    ];
    if (opts.includeModel) tasks.push(loadModel());

    var results = await Promise.allSettled(tasks);
    results.forEach(function (result) {
      if (result.status === 'fulfilled') {
        anySuccess = true;
      } else if (isOffline(result.reason)) {
        offlineSeen = true;
      }
    });

    reportConnectivity(offlineSeen, anySuccess);
    if (anySuccess) markUpdated();

    state.running = false;
    if (button) {
      button.disabled = false;
      button.removeAttribute('data-busy');
      button.textContent = 'Refresh';
    }
  }

  function stopTimer() {
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
  }

  function startTimer() {
    stopTimer();
    if (!state.intervalMs) return;
    state.timer = window.setInterval(function () {
      if (document.hidden) return; // no requests while the tab is in the background
      refreshAll();
    }, state.intervalMs);
  }

  /* ------------------------------------------------------------------ *
   * Events
   * ------------------------------------------------------------------ */

  function bindEvents() {
    var refreshBtn = $('refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', function () {
        refreshAll({ includeModel: true });
      });
    }

    var retryBtn = $('connection-retry');
    if (retryBtn) {
      retryBtn.addEventListener('click', function () {
        refreshAll({ includeModel: true });
      });
    }

    var interval = $('refresh-interval');
    if (interval) {
      interval.value = String(state.intervalMs);
      if (interval.value !== String(state.intervalMs)) {
        state.intervalMs = config.REFRESH_INTERVAL_MS || 30000;
        interval.value = String(state.intervalMs);
      }
      interval.addEventListener('change', function () {
        state.intervalMs = Number(interval.value) || 0;
        startTimer();
      });
    }

    var selector = $('location-select');
    if (selector) {
      selector.addEventListener('change', function () {
        selectLocation(selector.value || null);
      });
    }

    document.querySelectorAll('.segmented__btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var hours = Number(btn.getAttribute('data-hours'));
        if (!hours) return;
        state.hours = hours;
        document.querySelectorAll('.segmented__btn').forEach(function (other) {
          var active = other === btn;
          other.classList.toggle('is-active', active);
          if (active) other.setAttribute('aria-pressed', 'true');
          else other.removeAttribute('aria-pressed');
        });
        var loading = $('trend-loading');
        if (loading) {
          loading.hidden = false;
          loading.textContent = 'Loading analytics…';
        }
        loadTrends();
      });
    });

    // Retry buttons rendered inside state blocks.
    document.addEventListener('click', function (event) {
      var target = event.target.closest ? event.target.closest('[data-retry]') : null;
      if (!target) return;
      var which = target.getAttribute('data-retry');
      var loaders = {
        current: loadCurrent,
        locations: loadLocations,
        status: loadStatus,
        model: loadModel,
        prediction: loadPrediction,
        history: loadHistory,
        trends: loadTrends
      };
      if (loaders[which]) loaders[which]();
    });

    // Clickable rows and location cards select a location.
    document.addEventListener('click', function (event) {
      var row = event.target.closest ? event.target.closest('[data-location]') : null;
      if (!row) return;
      var id = row.getAttribute('data-location');
      if (id) selectLocation(id);
    });

    // Sidebar toggle (mobile) and section highlighting.
    var sidebarToggle = $('sidebar-toggle');
    var sidebar = $('dash-sidebar');
    if (sidebarToggle && sidebar) {
      sidebarToggle.addEventListener('click', function () {
        var open = sidebar.classList.toggle('is-open');
        sidebarToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
      sidebar.addEventListener('click', function (event) {
        if (event.target.tagName === 'A' && window.matchMedia('(max-width: 1080px)').matches) {
          sidebar.classList.remove('is-open');
          sidebarToggle.setAttribute('aria-expanded', 'false');
        }
      });
    }

    var sectionLinks = document.querySelectorAll('.dash-nav a[data-section]');
    sectionLinks.forEach(function (link) {
      link.addEventListener('click', function () {
        sectionLinks.forEach(function (other) {
          other.removeAttribute('aria-current');
        });
        link.setAttribute('aria-current', 'true');
      });
    });

    if ('IntersectionObserver' in window) {
      var observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (!entry.isIntersecting) return;
            sectionLinks.forEach(function (link) {
              if (link.getAttribute('data-section') === entry.target.id) {
                link.setAttribute('aria-current', 'true');
              } else {
                link.removeAttribute('aria-current');
              }
            });
          });
        },
        { rootMargin: '-25% 0px -65% 0px' }
      );
      document.querySelectorAll('.dash-section').forEach(function (section) {
        observer.observe(section);
      });
    }

    // Resume from a hidden tab only if the interval has elapsed.
    document.addEventListener('visibilitychange', function () {
      if (document.hidden || !state.intervalMs || !state.lastUpdated) return;
      if (Date.now() - state.lastUpdated.getTime() >= state.intervalMs) {
        refreshAll();
      }
    });

    window.addEventListener('resize', function () {
      window.TrafficMap.invalidate();
      window.TrafficCharts.invalidate();
    });
  }

  /* ------------------------------------------------------------------ *
   * Boot
   * ------------------------------------------------------------------ */

  function boot() {
    if (!API) return;

    UI.setLoading($('current-table'), 'Loading traffic…');
    UI.setLoading($('location-detail'), 'Loading traffic data…');
    UI.setLoading($('latest-prediction'), 'Loading predictions…');
    UI.setLoading($('prediction-history'), 'Loading predictions…');
    UI.setLoading($('locations-grid'), 'Loading locations…');
    UI.setLoading($('system-status'), 'Loading system status…');
    UI.setLoading($('model-info'), 'Loading analytics…');
    setConn('checking', 'Connecting…');

    window.TrafficMap.init('map', function (locationId) {
      selectLocation(locationId);
    });
    window.TrafficCharts.init('trend-canvas');

    bindEvents();
    resetPredictionPanels();
    refreshAll({ includeModel: true });
    startTimer();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
