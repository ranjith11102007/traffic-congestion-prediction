/**
 * Chart.js trend chart for GET /api/dashboard/trends.
 *
 * One chart, two speed series and the forecast the API attached to each point.
 * Points the API returned with `congestion_prediction: null` are drawn as gaps
 * (`spanGaps: false`), because drawing a line through them would invent a
 * forecast that was never made.
 *
 * The instance is destroyed before it is recreated, so a refresh can never
 * stack a second chart on the same canvas.
 */
(function () {
  'use strict';

  var UI = window.TrafficUI;

  var chart = null;
  var canvas = null;
  var libraryMissing = false;

  var PALETTE = {
    avg: '#3b82f6',
    free: '#94a3b8',
    prediction: '#f59e0b',
    grid: 'rgba(154, 171, 196, 0.16)',
    text: '#9aabc4'
  };

  function showNotice(message, detail) {
    if (!canvas) return;
    var host = canvas.closest('.chart-host') || canvas.parentElement;
    if (!host) return;
    var notice = host.querySelector('.chart-notice');
    if (!notice) {
      notice = document.createElement('div');
      notice.className = 'chart-notice';
      notice.setAttribute('role', 'status');
      host.appendChild(notice);
    }
    notice.innerHTML =
      '<strong>' + UI.esc(message) + '</strong>' +
      (detail ? '<span>' + UI.esc(detail) + '</span>' : '');
    notice.hidden = false;
  }

  function hideNotice() {
    if (!canvas) return;
    var host = canvas.closest('.chart-host') || canvas.parentElement;
    if (!host) return;
    var notice = host.querySelector('.chart-notice');
    if (notice) notice.hidden = true;
  }

  function labelsFor(points) {
    return points.map(function (point) {
      var date = new Date(point.timestamp);
      return isNaN(date.getTime()) ? '' : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });
  }

  function emptyState(points) {
    if (!points || !points.length) return true;
    var allNull = points.every(function (point) {
      return typeof point.avg_speed_kph !== 'number';
    });
    return allNull;
  }

  function buildDatasets(points) {
    return [
      {
        label: 'Average speed (km/h)',
        data: points.map(function (p) {
          return typeof p.avg_speed_kph === 'number' ? p.avg_speed_kph : null;
        }),
        borderColor: PALETTE.avg,
        backgroundColor: PALETTE.avg,
        borderWidth: 2,
        pointRadius: points.length > 80 ? 0 : 2,
        pointHoverRadius: 4,
        tension: 0.25,
        spanGaps: false,
        yAxisID: 'y'
      },
      {
        label: 'Free-flow speed (km/h)',
        data: points.map(function (p) {
          return typeof p.free_flow_speed_kph === 'number' ? p.free_flow_speed_kph : null;
        }),
        borderColor: PALETTE.free,
        backgroundColor: PALETTE.free,
        borderWidth: 1.5,
        borderDash: [6, 4],
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.25,
        spanGaps: false,
        yAxisID: 'y'
      },
      {
        label: 'Predicted congestion (class)',
        data: points.map(function (p) {
          return typeof p.congestion_level === 'number' ? p.congestion_level : null;
        }),
        borderColor: PALETTE.prediction,
        backgroundColor: PALETTE.prediction,
        borderWidth: 2,
        pointRadius: points.length > 80 ? 0 : 3,
        pointHoverRadius: 5,
        stepped: false,
        tension: 0.1,
        spanGaps: false,
        yAxisID: 'yCongestion'
      }
    ];
  }

  window.TrafficCharts = {
    /** @param {string} elementId canvas id inside a .chart-host container */
    init: function (elementId) {
      canvas = document.getElementById(elementId);
      if (!canvas) return false;
      if (typeof window.Chart === 'undefined') {
        libraryMissing = true;
        showNotice(
          'Chart library failed to load',
          'Chart.js is served from frontend/vendor. The table views still work.'
        );
        return false;
      }
      return true;
    },

    /**
     * Render a trends response, or explain why there is nothing to draw.
     *
     * @param {object} trends  response from GET /api/dashboard/trends
     * @param {{labelMap?: object}} options
     */
    render: function (trends, options) {
      if (!canvas) return;
      if (libraryMissing || typeof window.Chart === 'undefined') return;

      if (chart) {
        chart.destroy();
        chart = null;
      }

      var points = (trends && trends.points) || [];

      if (emptyState(points)) {
        showNotice(
          'No readings in this window',
          (trends && trends.note) ||
            'Collect observations for this location to build a trend. The chart will fill in as readings arrive.'
        );
        return;
      }
      hideNotice();

      var labelMap = (options && options.labelMap) || {};
      var ctx = canvas.getContext('2d');

      chart = new window.Chart(ctx, {
        type: 'line',
        data: {
          labels: labelsFor(points),
          datasets: buildDatasets(points)
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: PALETTE.text, boxWidth: 14, font: { size: 12 } }
            },
            tooltip: {
              callbacks: {
                afterBody: function (items) {
                  var index = items.length ? items[0].dataIndex : -1;
                  if (index < 0) return null;
                  var point = points[index];
                  var lines = [];
                  lines.push('Observed: ' + UI.formatDateTime(point.timestamp));
                  lines.push(
                    'Forecast: ' +
                      (point.congestion_prediction
                        ? UI.titleCase(point.congestion_prediction)
                        : 'none stored')
                  );
                  lines.push('Weather: ' + UI.titleCase(point.weather_condition));
                  lines.push('Source: ' + UI.sourceText(point.data_source, false));
                  return lines;
                }
              }
            }
          },
          scales: {
            x: {
              ticks: { color: PALETTE.text, maxRotation: 0, autoSkip: true, maxTicksLimit: 10 },
              grid: { color: PALETTE.grid }
            },
            y: {
              position: 'left',
              beginAtZero: true,
              title: { display: true, text: 'km/h', color: PALETTE.text },
              ticks: { color: PALETTE.text },
              grid: { color: PALETTE.grid }
            },
            yCongestion: {
              position: 'right',
              min: -0.5,
              max: 3.5,
              title: { display: true, text: 'Predicted class', color: PALETTE.text },
              ticks: {
                color: PALETTE.text,
                stepSize: 1,
                callback: function (value) {
                  return UI.congestionLabel(value, labelMap) || value;
                }
              },
              grid: { drawOnChartArea: false }
            }
          }
        }
      });
    },

    destroy: function () {
      if (chart) {
        chart.destroy();
        chart = null;
      }
    },

    invalidate: function () {
      if (chart) chart.resize();
    }
  };
})();
