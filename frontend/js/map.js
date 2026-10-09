/**
 * Leaflet map of the monitored locations.
 *
 * The map renders only what the API returns: coordinates come from
 * GET /api/dashboard/locations, and the marker colour comes from the forecast
 * on the matching location in GET /api/dashboard/current. A location with no
 * coordinate or no forecast is drawn neutrally rather than guessed at.
 *
 * Failure isolation: if the Leaflet library or the raster tiles fail to load,
 * this module reports it inside the map card and nothing else on the page is
 * affected.
 */
(function () {
  'use strict';

  var config = window.TRAFFIC_CONFIG || {};
  var UI = window.TrafficUI;

  var map = null;
  var tileLayer = null;
  var markers = {}; // location_id -> L.circleMarker
  var container = null;
  var onSelect = null;
  var selectedId = null;
  var libraryMissing = false;

  var TONE_COLOURS = {
    low: '#22c55e',
    moderate: '#f59e0b',
    high: '#f97316',
    severe: '#ef4444',
    unknown: '#94a3b8'
  };

  function showNotice(title, detail) {
    if (!container) return;
    var notice = container.querySelector('.map-notice');
    if (!notice) {
      notice = document.createElement('div');
      notice.className = 'map-notice';
      notice.setAttribute('role', 'status');
      container.appendChild(notice);
    }
    notice.innerHTML =
      '<strong>' + UI.esc(title) + '</strong>' +
      (detail ? '<span>' + UI.esc(detail) + '</span>' : '');
    notice.hidden = false;
  }

  function hideNotice() {
    if (!container) return;
    var notice = container.querySelector('.map-notice');
    if (notice) notice.hidden = true;
  }

  function popupHtml(location, current) {
    if (!current) {
      return (
        '<div class="map-popup">' +
        '<h4>' + UI.esc(location.road_name || location.location_id) + '</h4>' +
        '<p class="map-popup__id">' + UI.esc(location.location_id) + '</p>' +
        '<p>No reading has been collected for this location.</p>' +
        '</div>'
      );
    }

    var prediction = current.prediction;
    var rows = [
      ['Location', UI.titleCase(location.location_id) === UI.NA ? location.location_id : location.location_id],
      ['Speed', UI.formatNumber(current.avg_speed_kph) + (current.avg_speed_kph !== null ? ' km/h' : '')],
      ['Free flow', UI.formatNumber(current.free_flow_speed_kph) + (current.free_flow_speed_kph !== null ? ' km/h' : '')]
    ];

    if (prediction) {
      rows.push(['Predicted', UI.titleCase(prediction.predicted_congestion)]);
      rows.push(['Confidence', UI.formatConfidence(prediction.confidence)]);
      rows.push(['Forecast made', UI.formatDateTime(prediction.prediction_timestamp)]);
    } else {
      rows.push(['Predicted', UI.predictionStatusText(current.prediction_status)]);
    }

    rows.push(['Temperature', UI.formatNumber(current.temperature_c) + (current.temperature_c !== null ? ' \u00B0C' : '')]);
    rows.push(['Weather', UI.titleCase(current.weather_condition)]);
    rows.push(['Observed', UI.formatDateTime(current.observation_timestamp)]);
    rows.push(['Freshness', UI.freshnessText(current.freshness)]);
    rows.push(['Source', UI.sourceText(current.data_source, current.is_simulation)]);

    var body = rows
      .map(function (row) {
        return (
          '<tr><th scope="row">' + UI.esc(row[0]) + '</th>' +
          '<td>' + UI.esc(row[1]) + '</td></tr>'
        );
      })
      .join('');

    return (
      '<div class="map-popup">' +
      '<h4>' + UI.esc(current.road_name || location.road_name) + '</h4>' +
      '<table>' + body + '</table>' +
      '</div>'
    );
  }

  function ensureMap() {
    if (map || libraryMissing) return map;
    if (typeof window.L === 'undefined') {
      libraryMissing = true;
      showNotice(
        'Map library failed to load',
        'Leaflet is served from frontend/vendor. Every other panel still works.'
      );
      return null;
    }

    container.innerHTML = '';
    map = window.L.map(container, {
      scrollWheelZoom: false,
      attributionControl: true
    }).setView([20, 0], 2);

    tileLayer = window.L.tileLayer(config.MAP_TILE_URL, {
      maxZoom: config.MAP_MAX_ZOOM || 19,
      minZoom: config.MAP_MIN_ZOOM || 3,
      attribution: config.MAP_TILE_ATTRIBUTION || '',
      // A failed tile leaves its cell blank instead of breaking the layer.
      errorTileUrl: ''
    });

    tileLayer.on('tileerror', function () {
      showNotice(
        'Map tiles are unavailable',
        'The OpenStreetMap tile server did not respond. Location markers and the rest of the dashboard are unaffected.'
      );
    });
    tileLayer.on('tileload', function () {
      hideNotice();
    });
    tileLayer.addTo(map);
    return map;
  }

  function currentById(locations) {
    var index = {};
    (locations || []).forEach(function (entry) {
      index[entry.location_id] = entry;
    });
    return index;
  }

  function focus(locationId) {
    selectedId = locationId || null;
    Object.keys(markers).forEach(function (id) {
      var marker = markers[id];
      var selected = id === selectedId;
      marker.setStyle({ weight: selected ? 3 : 1, color: selected ? '#e6edf7' : '#0b1220' });
      if (selected) marker.bringToFront();
    });
  }

  window.TrafficMap = {
    /**
     * @param {string} elementId  id of the DOM node to host the map
     * @param {(locationId: string) => void} selectCallback marker click
     */
    init: function (elementId, selectCallback) {
      container = document.getElementById(elementId);
      onSelect = selectCallback || null;
      if (!container) return false;
      ensureMap();
      return true;
    },

    /**
     * Replace every marker from API data. Nothing is kept from a previous
     * poll, so a location that disappears from the API disappears from the map.
     *
     * @param {Array} locations  from GET /api/dashboard/locations
     * @param {Array} currents   from GET /api/dashboard/current
     */
    update: function (locations, currents) {
      if (!ensureMap()) return;

      var index = currentById(currents);
      var bounds = [];

      Object.keys(markers).forEach(function (id) {
        if (map.hasLayer(markers[id])) map.removeLayer(markers[id]);
      });
      markers = {};

      (locations || []).forEach(function (location) {
        if (
          typeof location.latitude !== 'number' ||
          typeof location.longitude !== 'number'
        ) {
          return;
        }

        var current = index[location.location_id] || null;
        var level = current && current.prediction
          ? current.prediction.predicted_congestion_level
          : null;
        var tone = UI.congestionTone(level);
        var hasReading = !!(current && current.avg_speed_kph !== null);

        var marker = window.L.circleMarker([location.latitude, location.longitude], {
          radius: 9,
          color: '#0b1220',
          weight: 1,
          fillColor: hasReading ? TONE_COLOURS[tone] : TONE_COLOURS.unknown,
          fillOpacity: hasReading ? 0.95 : 0.45
        });

        marker.bindPopup(popupHtml(location, current), { maxWidth: 320 });
        marker.on('click', function () {
          selectedId = location.location_id;
          if (onSelect) onSelect(location.location_id);
        });

        marker.addTo(map);
        markers[location.location_id] = marker;
        bounds.push([location.latitude, location.longitude]);
      });

      if (bounds.length === 1) {
        map.setView(bounds[0], 13);
      } else if (bounds.length > 1) {
        map.fitBounds(bounds, { padding: [30, 30], maxZoom: 13 });
      }

      focus(selectedId);
      hideNotice();
    },

    /** Re-centre on one location, e.g. after a selector change. */
    select: function (locationId) {
      selectedId = locationId || null;
      var marker = markers[selectedId];
      if (marker && map) {
        map.setView(marker.getLatLng(), Math.max(map.getZoom(), 13));
        marker.openPopup();
      }
      focus(selectedId);
    },

    /** Let the map recalculate its size when its container changes size. */
    invalidate: function () {
      if (map) map.invalidateSize();
    }
  };
})();
