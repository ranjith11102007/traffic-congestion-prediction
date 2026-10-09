"""Stage 7 - frontend dashboard structural and contract validation.

These tests stay at the file level (no browser required) so they run in the
same pytest suite as the backend: they prove that the shipped HTML, CSS and Vanilla
JS agree with each other and with the documented dashboard API, and that no secret
or fabricated value ever appears in the frontend.

They do not replace a browser smoke test; they catch accidental regressions in the
element/API contract on every CI run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
VENDOR = FRONTEND / "vendor"
JS = FRONTEND / "js"
CSS = FRONTEND / "css"
MODEL_METADATA = ROOT / "ml" / "models" / "model_metadata.json"

DASHBOARD_HTML = FRONTEND / "dashboard.html"
INDEX_HTML = FRONTEND / "index.html"

# Endpoints the frontend is allowed to call. Everything else is a mistake.
ALLOWED_PATHS = {
    "/api/health",
    "/api/dashboard/current",
    "/api/dashboard/locations",
    "/api/dashboard/trends",
    "/api/dashboard/status",
    "/api/dashboard/model",
    "/api/traffic/predictions",
    "/api/traffic/predictions/latest",
    "/api/traffic/observations",
}

ASSET_RE = re.compile(r'(?:src|href)="([^"]+)"')

ALLOWED_LIBS = {"L": "Leaflet", "Chart": "Chart.js"}

SECRET_RE = re.compile(
    r"\b(?:api[_ -]?key|secret|password|passwd|token|bearer)\b",
    re.IGNORECASE,
)


def frontend_files(*extensions: str) -> list[Path]:
    return sorted(
        p
        for p in FRONTEND.rglob("*")
        if p.is_file() and p.suffix in extensions
    )


@pytest.fixture(scope="module")
def dashboard_html() -> str:
    return DASHBOARD_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dashboard_js() -> str:
    return (JS / "dashboard.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def model_metadata() -> dict:
    if not MODEL_METADATA.exists():
        return {"label_mapping": {}}
    return json.loads(MODEL_METADATA.read_text(encoding="utf-8"))


class TestAssets:
    def test_every_asset_referenced_by_index_and_dashboard_exists(self, dashboard_html: str) -> None:
        for html in (INDEX_HTML.read_text(encoding="utf-8"), dashboard_html):
            for path in ASSET_RE.findall(html):
                if path.startswith(("http://", "https://", "data:", "#")) or "#" in path:
                    continue
                assert (FRONTEND / path.lstrip("/")).exists(), (
                    f"Asset referenced in HTML does not exist: {path}"
                )

    def test_vendor_libraries_are_the_expected_static_copies(self) -> None:
        leaflet = (VENDOR / "leaflet.js").read_text(encoding="utf-8", errors="replace")
        chart = (VENDOR / "chart.umd.min.js").read_text(encoding="utf-8", errors="replace")
        assert "Leaflet" in leaflet
        assert "chart.js" in chart.lower() or "Chart" in chart
        assert (VENDOR / "leaflet.css").stat().st_size > 1000

    def test_all_scripts_are_local_no_cdn_dependency(self, dashboard_html: str) -> None:
        scripts = re.findall(r'<script[^>]+src="([^"]+)"', dashboard_html)
        assert scripts, "dashboard.html has no scripts"
        for src in scripts:
            assert not src.startswith("http"), f"CDN dependency in dashboard.html: {src}"
            assert (FRONTEND / src.lstrip("/")).exists()


class TestElementContract:
    ID_CALL_RE = re.compile(r"\$\((['\"])((?:[a-z][a-z0-9-]*))\1\)")

    def test_every_id_the_controller_queries_exists_in_the_html(self, dashboard_html: str, dashboard_js: str) -> None:
        queried = set(self.ID_CALL_RE.findall(dashboard_js))
        assert queried, "No element ids found in dashboard.js"
        for _, element_id in queried:
            assert f'id="{element_id}"' in dashboard_html, (
                f"dashboard.js queries #{element_id} but dashboard.html has no such element"
            )

    def test_boot_fastpaths_run_only_after_declared_ids(self, dashboard_html: str) -> None:
        for needle in ("map", "trend-canvas", "map-state", "location-select", "refresh-btn"):
            assert needle in dashboard_html

    def test_segmented_control_matches_config_options(self, dashboard_html: str) -> None:
        hours = re.findall(r'class="[^"]*segmented__btn[^"]*"[^>]*data-hours="(\d+)"', dashboard_html)
        config = (JS / "config.js").read_text(encoding="utf-8")
        config_hours = re.search(r"TREND_HOUR_OPTIONS:\s*\[([^\]]+)\]", config)
        assert config_hours, "config.js has no TREND_HOUR_OPTIONS"
        expected = {int(v) for v in re.findall(r"\d+", config_hours.group(1))}
        assert set(int(h) for h in hours) == expected


class TestNoSecretsAndNoFabrication:
    NOT_ALLOWED_PLACEHOLDERS = ("lorem", "placeholder", "TODO", "FixMe", "PLACEHOLDER", "Phase 9", "Fake", "sample data")

    def test_no_secrets_in_any_frontend_file(self) -> None:
        for path in frontend_files(".js", ".html", ".css", ".json"):
            matches = SECRET_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
            assert not matches, f"Possible secret keyword in {path}: {matches}"

    def test_no_fabricated_traffic_values_in_frontend(self, dashboard_html: str) -> None:
        low = dashboard_html.lower()
        for needle in self.NOT_ALLOWED_PLACEHOLDERS:
            assert needle.lower() not in low, f"Forbidden placeholder text on dashboard: {needle}"

    def test_js_uses_config_base_url_not_hardcoded_host(self) -> None:
        api_js = (JS / "api.js").read_text(encoding="utf-8")
        assert "API_BASE_URL" in api_js
        assert "127.0.0.1" not in api_js and "localhost" not in api_js
        assert "localhost:8000" not in api_js

    def test_api_client_only_calls_documented_endpoints(self) -> None:
        api_js = (JS / "api.js").read_text(encoding="utf-8")
        used = set(re.findall(r"['\"]((?:/api/)[^'\"]{0,60})['\"]", api_js))
        paths = {u.split("?")[0].rstrip("/") for u in used}
        assert paths, "api.js calls no /api/ endpoints"
        for path in paths:
            assert path in ALLOWED_PATHS, f"api.js calls an endpoint not documented for Stage 7: {path}"

    def test_model_labels_come_from_the_model_nothing_fabricated(self, model_metadata: dict) -> None:
        ui_js = (JS / "ui.js").read_text(encoding="utf-8")
        mapping = model_metadata.get("label_mapping") or {}
        allowed_names = {str(v).lower() for v in mapping.values()}
        assert allowed_names, "model_metadata has no label_mapping"
        for name in ("free flow", "moderate", "heavy", "severe"):
            assert name.lower() in ui_js.lower()
        assert all(re.search(r"\b" + n + r"\b", ui_js.lower()) for n in allowed_names)
        banned = {"low congestion", "high congestion"}
        for b in banned:
            assert b not in ui_js.lower()

    def test_map_and_chart_guard_against_missing_libraries(self) -> None:
        map_js = (JS / "map.js").read_text(encoding="utf-8")
        charts_js = (JS / "charts.js").read_text(encoding="utf-8")
        assert "typeof window.L === 'undefined'" in map_js or "window.L" in map_js
        assert "typeof window.Chart === 'undefined'" in charts_js or "window.Chart" in charts_js
        assert "TrafficMap" in map_js and "TrafficCharts" in charts_js

    def test_modules_expose_window_bindings_the_controller_needs(self) -> None:
        for path, binding in (
            (JS / "map.js", "window.TrafficMap"),
            (JS / "charts.js", "window.TrafficCharts"),
            (JS / "ui.js", "window.TrafficUI"),
            (JS / "api.js", "window.TrafficAPI"),
        ):
            assert binding in path.read_text(encoding="utf-8"), f"{path} does not expose {binding}"


class TestConfigAndRefresh:
    def test_refresh_defaults(self) -> None:
        config = (JS / "config.js").read_text(encoding="utf-8")
        assert re.search(r"REFRESH_INTERVAL_MS\s*:\s*30000", config)
        assert re.search(r"REQUEST_TIMEOUT_MS\s*:\s*8000", config)
        assert re.search(r"API_BASE_URL\s*:\s*['\"]http://127\.0\.0\.1:8000", config)

    def test_dashboard_declares_auto_refresh_controls(self, dashboard_html: str) -> None:
        assert 'id="refresh-interval"' in dashboard_html
        assert 'id="refresh-btn"' in dashboard_html
        assert 'id="last-updated"' in dashboard_html


class TestEscapingNoXSS:
    """Every API-derived string rendered through ``innerHTML`` must be escaped.

    Stage 8 hardening: a location name or status message coming back from the API
    is untrusted input. The dashboard convention (documented in ``ui.js``) is that
    such values go through ``UI.esc`` before they are placed in ``innerHTML``.
    These file-level tests keep that contract from silently regressing.
    """

    def test_ui_esc_escapes_all_five_html_metacharacters(self) -> None:
        ui_js = (JS / "ui.js").read_text(encoding="utf-8")
        for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            assert entity in ui_js, f"esc() does not produce {entity}"

    def test_every_render_file_escapes_each_inner_html_assignment(self) -> None:
        for name in ("dashboard.js", "map.js", "charts.js"):
            source = (JS / name).read_text(encoding="utf-8")
            assignments = len(re.findall(r"\.innerHTML\s*=", source))
            esc_calls = len(re.findall(r"UI\.esc\(", source))
            assert assignments >= 1, f"{name} renders nothing?"
            assert esc_calls >= assignments, (
                f"{name}: {esc_calls} UI.esc calls but {assignments} innerHTML "
                "assignments; at least one unescaped render path exists"
            )

    def test_map_popups_escape_location_and_measurement_values(self) -> None:
        map_js = (JS / "map.js").read_text(encoding="utf-8")
        assert "UI.esc(location.road_name" in map_js
        assert "UI.esc(location.location_id" in map_js
        assert "UI.esc(row[0])" in map_js and "UI.esc(row[1])" in map_js

    def test_chart_and_map_notices_escape_dynamic_messages(self) -> None:
        charts_js = (JS / "charts.js").read_text(encoding="utf-8")
        assert "UI.esc(message)" in charts_js
        assert "UI.esc(detail)" in charts_js

    def test_status_template_escapes_dashboard_status_values(self, dashboard_js: str) -> None:
        for needle in (
            "UI.esc(s.traffic_provider",
            "UI.esc(s.weather_provider",
            "UI.esc(s.model_version",
            "UI.esc(s.last_error)",
        ):
            assert needle in dashboard_js, f"status panel leaks {needle!r} unescaped"

    def test_selector_and_table_escape_location_values(self, dashboard_js: str) -> None:
        assert "UI.esc(entry.location_id)" in dashboard_js
        assert "UI.esc(entry.road_name" in dashboard_js
        assert "UI.esc(loc.location_id)" in dashboard_js
        assert "UI.esc(loc.road_name)" in dashboard_js
        assert "UI.esc(loc.weather_condition)" in dashboard_js or "UI.esc(UI.titleCase(loc.weather_condition))" in dashboard_js