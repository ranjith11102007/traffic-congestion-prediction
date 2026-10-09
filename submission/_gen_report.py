"""Generate submission/AI_Traffic_Congestion_Prediction_Project_Report.pdf.

All content is drawn from the verified project documentation
(docs/project-report.md, README.md, docs/architecture.md, docs/deployment.md).
No numbers, claims or identifiers are invented.
"""

from __future__ import annotations

import html
import unicodedata

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

PDF_PATH = r"C:\Users\RANJITH T\Desktop\TV\submission\AI_Traffic_Congestion_Prediction_Project_Report.pdf"

# --- Fonts -------------------------------------------------------------------
FDIR = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Calibri", f"{FDIR}\\calibri.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-Bold", f"{FDIR}\\calibrib.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-Italic", f"{FDIR}\\calibrii.ttf"))
pdfmetrics.registerFont(TTFont("Consolas", f"{FDIR}\\consola.ttf"))

NAVY = colors.HexColor("#1F3864")
STEEL = colors.HexColor("#2E5C9E")
LIGHT = colors.HexColor("#F2F6FB")
GRID = colors.HexColor("#C9D4E4")
GREY = colors.HexColor("#595959")


def sanitize(text: str) -> str:
    """Convert unicode arrows/dashes to ASCII and XML-escape for reportlab."""
    table = {
        "\u2192": "->",
        "\u2190": "<-",
        "\u2194": "<->",
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b7": "-",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00d7": "x",
    }
    for src, dst in table.items():
        text = text.replace(src, dst)
    return html.escape(text, quote=False)


def P(text: str, style) -> Paragraph:
    return Paragraph(sanitize(text), style)


def CODE(text: str, style) -> Preformatted:
    return Preformatted(sanitize(text), style)


# --- Styles -------------------------------------------------------------------
def build_styles():
    base = getSampleStyleSheet()
    S = {}
    S["H1"] = ParagraphStyle(
        "H1", parent=base["Heading1"], fontName="Calibri-Bold", fontSize=15,
        leading=18, textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=1,
    )
    S["H2"] = ParagraphStyle(
        "H2", parent=base["Heading2"], fontName="Calibri-Bold", fontSize=12,
        leading=15, textColor=STEEL, spaceBefore=10, spaceAfter=4, keepWithNext=1,
    )
    S["Body"] = ParagraphStyle(
        "Body", parent=base["Normal"], fontName="Calibri", fontSize=10,
        leading=13.8, alignment=TA_JUSTIFY, spaceAfter=6,
    )
    S["Bullet"] = ParagraphStyle(
        "Bullet", parent=S["Body"], leftIndent=16, bulletIndent=4, spaceAfter=3,
    )
    S["Code"] = ParagraphStyle(
        "Code", parent=base["Code"], fontName="Consolas", fontSize=7.5,
        leading=9.6, textColor=colors.HexColor("#1E1E1E"), backColor=colors.HexColor("#F4F4F4"),
        borderPadding=6, spaceBefore=4, spaceAfter=8,
    )
    S["Caption"] = ParagraphStyle(
        "Caption", parent=S["Body"], fontName="Calibri-Italic", fontSize=8.5,
        alignment=TA_CENTER, textColor=GREY, spaceBefore=2, spaceAfter=8,
    )
    S["TblCell"] = ParagraphStyle(
        "TblCell", parent=base["Normal"], fontName="Calibri", fontSize=9,
        leading=11.5,
    )
    S["TblHead"] = ParagraphStyle(
        "TblHead", parent=S["TblCell"], fontName="Calibri-Bold", textColor=colors.white,
    )
    S["CoverTitle"] = ParagraphStyle(
        "CoverTitle", parent=base["Normal"], fontName="Calibri-Bold", fontSize=23,
        leading=28, alignment=TA_CENTER, textColor=NAVY, spaceAfter=8,
    )
    S["CoverSub"] = ParagraphStyle(
        "CoverSub", parent=S["CoverTitle"], fontName="Calibri", fontSize=13,
        leading=17, textColor=GREY,
    )
    return S


def hline(width, color=GRID, thickness=0.8):
    tbl = Table([[""]], colWidths=[width], rowHeights=[2])
    tbl.setStyle(TableStyle([("LINEABOVE", (0, 0), (0, 0), thickness, color)]))
    return tbl


S = build_styles()


# --- Architecture diagram ----------------------------------------------
def architecture_diagram() -> Drawing:
    import math

    d = Drawing(483, 340)
    h = 340
    fill = LIGHT

    def box(x, y, wdt, hgt, title, lines, tc=NAVY, fillc=fill, tsize=9):
        d.add(Rect(x, y, wdt, hgt, rx=6, ry=6, fillColor=fillc,
                   strokeColor=tc, strokeWidth=1.1))
        d.add(String(x + wdt / 2.0, y + hgt - 13, title, fontName="Calibri-Bold",
                     fontSize=tsize, fillColor=tc, textAnchor="middle"))
        yy = y + hgt - 27
        for line in lines:
            d.add(String(x + wdt / 2.0, yy, line, fontName="Calibri", fontSize=8,
                         fillColor=colors.HexColor("#333333"), textAnchor="middle"))
            yy -= 11

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=STEEL, strokeWidth=1.4))
        ang = math.atan2(y2 - y1, x2 - x1)
        L, A = 7, 0.44
        d.add(Polygon([x2, y2, x2 - L * math.cos(ang - A), y2 - L * math.sin(ang - A),
                       x2 - L * math.cos(ang + A), y2 - L * math.sin(ang + A)],
                      fillColor=STEEL, strokeColor=STEEL))

    # Browser / frontend (top centre)
    bw, tw = 200, 260
    bx = (483 - bw) / 2.0
    box(bx, 276, bw, 56, "Browser / Frontend",
        ["static HTML5 + CSS3 + JS - Leaflet map", "Chart.js analytics - auto-refresh"])
    arrow(483 / 2.0, 276, 483 / 2.0, 262)
    # FastAPI backend
    tx = 112
    box(tx, 180, tw, 80, "FastAPI backend",
        ["api/ routers - schemas/ validation", "services/ logic - repositories/ SQL",
         "one error envelope - CORS"])
    # Providers fed into the backend
    box(8, 180, 96, 80, "Data providers",
        ["Simulated", "(labelled)", "Real: TomTom", "Weather: Open-Meteo"],
        tc=colors.HexColor("#8A6D1D"), tsize=8.5)
    arrow(104, 220, tx, 220)
    d.add(String(56, 164, "opt-in scheduler:  collect -> predict", fontName="Calibri-Italic",
                 fontSize=7.5, fillColor=GREY))
    # Authoritative drop into the persistence / ML row
    arrow(483 / 2.0, 180, 130, 158)
    arrow(483 / 2.0, 180, 358, 158)
    # ML and PostgreSQL boxes (bottom row)
    box(20, 80, 210, 72, "ML pipeline (ml/)",
        ["Random Forest v4.0.0 - 26 features", "offline train / evaluate / predict",
         "artifacts with provenance"])
    box(253, 80, 210, 72, "PostgreSQL 14+",
        ["traffic_observations", "traffic_predictions", "Alembic migrations"])
    return d


# --- Content ------------------------------------------------------------------
N = "____________________"

ident_rows = [
    ("Project Title", "AI-Based Urban Traffic Congestion Prediction System"),
    ("Student Name", N),
    ("Registration / Roll No.", N),
    ("Programme / Branch", N),
    ("Institution / College", N),
    ("Project Guide / Supervisor", N),
    ("Submission Year", "20____"),
    ("Project Status", "Stages 1-8 complete; 450 automated tests passing"),
]

stack_rows = [
    ("Layer", "Technology", "Notes"),
    ("Frontend", "HTML5, CSS3, Vanilla JavaScript (ES2020)", "No build step; Leaflet 1.9.4 + Chart.js 4.4.1 vendored locally"),
    ("Backend", "Python 3.11+, FastAPI, Pydantic v2, Uvicorn", "Application-factory pattern; typed schemas"),
    ("Database", "PostgreSQL 14+, SQLAlchemy 2.x, psycopg2", "Env-driven lazy engine; Alembic migrations"),
    ("ML", "NumPy, pandas, scikit-learn, joblib", "Independent package; artifact bundles with provenance"),
    ("API client", "httpx", "Also drives mocked provider contract tests"),
    ("Testing", "pytest, httpx.MockTransport", "450 tests; isolated SQLite; jsdom browser simulation"),
]

tech_rows = [
    ("Area", "Technology used"),
    ("Backend framework", "FastAPI (Python 3.11+)"),
    ("Validation / schemas", "Pydantic v2"),
    ("Web server", "Uvicorn"),
    ("Database", "PostgreSQL 14+, SQLAlchemy 2.x, psycopg2"),
    ("Migrations", "Alembic (3-revision chain)"),
    ("Machine learning", "scikit-learn RandomForestClassifier"),
    ("Data handling", "NumPy, pandas, joblib"),
    ("External data", "TomTom Flow Segment Data API; Open-Meteo forecast API (httpx)"),
    ("Frontend", "HTML5, CSS3, Vanilla JS; Leaflet.js maps; Chart.js"),
    ("Deployment / config", "Environment-driven (.env); Uvicorn; reverse-proxy ready"),
    ("Testing", "pytest (+ jsdom browser simulation)"),
]

api_rows = [
    ("Method", "Path", "Purpose"),
    ("GET", "/api/health", "Liveness probe"),
    ("POST / GET", "/api/traffic/observations[/{id}]", "Record / list / fetch readings"),
    ("GET", "/api/traffic/latest", "Most recent reading"),
    ("POST", "/api/traffic/predict", "Store a reading, predict it, persist the forecast"),
    ("POST", "/api/traffic/observations/{id}/predict", "Predict from a stored reading"),
    ("GET", "/api/traffic/predictions[/latest][/status]", "History, latest forecast, integration status"),
    ("GET", "/api/traffic/collector/status", "Active provider + simulated flag"),
    ("POST", "/api/traffic/collect[/history]", "Collect -> auto-predict / backfill history"),
    ("GET", "/api/dashboard/current | trends | locations | status | model", "Single-poll dashboard reads"),
]

db_rows = [
    ("Table", "Purpose", "Key constraints"),
    ("traffic_observations", "One reading per road per time", "Unique (location_id, timestamp); avg_speed_kph <= free_flow_speed_kph; nullable vehicle_count; source provenance"),
    ("traffic_predictions", "One forecast per observation", "FK ON DELETE CASCADE; predicted_congestion_level in 0-3; confidence nullable"),
]

migration_rows = [
    ("Item", "Method", "Result"),
    ("Migration round-trip", "alembic upgrade head -> downgrade base -> upgrade head (fresh SQLite)", "Reach head c3d4e5f6a7b8; 2 tables, 12 indexes, unique constraint; re-applied cleanly"),
    ("PostgreSQL dialect", "Offline --sql rendering + migration tests", "Dialect-correct DDL verified"),
    ("Live PostgreSQL round-trip", "Not possible here (no server)", "Procedure documented; plain alembic upgrade head"),
]

test_rows = [
    ("Item", "Count / coverage"),
    ("Backend API / domain tests", "~422 tests at Stage 7 close"),
    ("Stage 7 frontend contract tests", "15 tests (structure, element contract, no secrets)"),
    ("Stage 8 additions", "7 production-guard config tests + 6 frontend XSS/escaping tests"),
    ("Total (final run)", "450 tests, all passing (plus 2 sklearn warnings)"),
    ("Compile checks", "python -m compileall -q (backend, ml, migrations, tests); node --check on every frontend/js/*.js"),
]

verify_rows = [
    ("Verification", "Outcome"),
    ("Automated test suite", "450 passed"),
    ("Open-Meteo live call", "HTTP 200; current.temperature_2m / precipitation / weather_code present"),
    ("TomTom live call", "NOT performed - no TRAFFIC_API_KEY available; covered by mocked contract tests"),
    ("SQLite Alembic round-trip", "upgrade -> downgrade -> re-upgrade verified"),
    ("PostgreSQL dialect", "Verified offline (SQL rendering)"),
    ("Model", "RandomForestClassifier v4.0.0, 26 features, 4 classes, dataset_is_simulated=true"),
    ("Production guard", "ENVIRONMENT=production boots only with real provider + key + PostgreSQL"),
    ("Frontend safety", "UI.esc covers & < > \" ' on every innerHTML template"),
]

results_rows = [
    ("Result", "Value"),
    ("Automated tests", "450 passed (0 failures)"),
    ("Model version", "4.0.0 (RandomForestClassifier, 26 features)"),
    ("Train / test split", "Chronological 80/20 (leakage-safe)"),
    ("Macro F1 (simulated fixture)", "0.7370"),
    ("Accuracy (simulated fixture)", "0.8561"),
    ("Majority-class baseline", "F1 0.2143 / accuracy 0.7500"),
    ("Live weather verification", "Open-Meteo HTTP 200"),
    ("Migration verification", "SQLite round-trip + offline PostgreSQL DDL"),
]


def bullets(items, style="Bullet"):
    return [Paragraph(sanitize(t), S[style], bulletText="\u2022") for t in items]


def make_table(rows, widths, head_style="TblHead", body_style="TblCell"):
    data = []
    for ri, row in enumerate(rows):
        line = []
        for cell in row:
            st = head_style if ri == 0 else body_style
            line.append(Paragraph(sanitize(cell), S[st]))
        data.append(line)
    tbl = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for ri in range(1, len(rows)):
        if ri % 2 == 0:
            style.append(("BACKGROUND", (0, ri), (-1, ri), LIGHT))
    tbl.setStyle(TableStyle(style))
    return tbl


def cover(story, styles):
    story.append(Spacer(1, 18))
    story.append(P("AI-Based Urban Traffic Congestion Prediction System",
                   styles["CoverTitle"]))
    story.append(P("Project Report | College / Internship Submission", styles["CoverSub"]))
    story.append(Spacer(1, 10))
    story.append(hline(340))
    story.append(Spacer(1, 28))
    story.append(make_table(ident_rows, [150, 300]))
    story.append(Spacer(1, 24))
    story.append(P("Prepared by", styles["CoverSub"]))
    story.append(Spacer(1, 6))
    story.append(P("Name: " + N + "        Roll No.: " + N, styles["Body"]))
    story.append(Spacer(1, 40))
    story.append(P("This report documents a verified software build (Stages 1-8). "
                   "All figures are execution results from the repository; no accuracy "
                   "claims for real-world traffic are made.", styles["Body"]))
    story.append(Spacer(1, 8))
    story.append(P("The enclosed forecasts are demonstrative until the model is "
                   "retrained and validated against real-world ground truth.",
                   styles["Body"]))
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())


def body(story, styles):
    W = 483

    def H1(t):
        story.append(P(t, styles["H1"]))

    def H2(t):
        story.append(P(t, styles["H2"]))

    def bodyp(t):
        story.append(P(t, styles["Body"]))

    # --- Abstract -----------------------------------------------------------------
    H1("Abstract")
    bodyp(
        "Urban traffic congestion is discovered too late: commuters learn about it "
        "once they are inside it, and traffic management reacts after congestion has "
        "already formed. This project builds a modular AI-based system that ingests "
        "urban traffic observations, engineers leakage-safe features, trains a "
        "machine-learning model to predict congestion level per road segment, and "
        "serves current state and near-horizon forecasts through a documented REST "
        "API and an interactive map dashboard."
    )
    bodyp(
        "The implementation spans eight stages that are each verified end to end: a "
        "six-stage data pipeline, a Random Forest classifier (version 4.0.0, 26 "
        "features) evaluated against an honest majority-class baseline, PostgreSQL "
        "persistence with Alembic migrations, a real-time ingestion path through the "
        "TomTom traffic API and the Open-Meteo weather API behind a provider "
        "interface, a background collection scheduler, a dashboard aggregation API, "
        "a static front-end dashboard, and a final hardening pass (production "
        "configuration guard, security review, deployment guide and this report). "
        "The complete build is verified by 450 automated tests."
    )
    bodyp(
        "Honesty is treated as a first-class requirement: the dataset used to train "
        "the model is a clearly labelled simulated fixture, so every forecast is "
        "demonstrative rather than field-validated. Connecting a live traffic "
        "provider does not change that. The system is designed so the same, tested "
        "pipeline can be retrained and revalidated on real observations once "
        "sufficient history is collected."
    )

    # --- Problem statement ----------------------------------------------------------
    H1("Problem Statement")
    bodyp(
        "Urban road networks experience congestion that people only discover once "
        "they are already inside it. Reactive traffic control responds after "
        "congestion has formed, and static timetables cannot account for the "
        "day-to-day variation caused by incidents, weather, construction and "
        "shifting peak hours. The consequences are measurable: lost travel time, "
        "wasted fuel, avoidable emissions, and unreliable travel planning."
    )
    bodyp(
        "Existing dashboards mostly describe current conditions. They answer 'where "
        "is it congested right now?' but not 'will this road be congested in 30 "
        "minutes?' - which is the question that would actually let someone change "
        "their behaviour. Additionally, congestion labels are usually produced by "
        "simple threshold rules; a severe-but-rare class is easily ignored by an "
        "accuracy-optimising approach, which is why this project evaluates with "
        "macro F1 against a majority-class baseline."
    )

    # --- Objectives -----------------------------------------------------------------
    H1("Objectives")
    story.extend(bullets([
        "Collect urban traffic observations into a persistent store, from a live provider when a traffic API key is configured and from a clearly labelled simulation otherwise.",
        "Engineer leakage-safe features from time, road geometry, traffic flow and weather.",
        "Train and evaluate a machine-learning classifier against a honest majority-class baseline on a chronological split.",
        "Expose predictions through a documented REST API with a uniform error envelope.",
        "Visualise current and predicted congestion on an interactive map dashboard.",
        "Harden the result for rollout: production configuration cannot silently fall back to simulated data, and everything is verifiable by automated tests.",
    ]))

    # --- Existing System -------------------------------------------------------------
    H1("Existing System")
    bodyp(
        "The context this project addresses is served today by systems that are "
        "largely descriptive and reactive:"
    )
    story.extend(bullets([
        "Live condition dashboards and traffic apps report current congestion, not near-term forecasts.",
        "Traffic control reacts after congestion appears; there is no forward-looking trigger.",
        "Static timetable and routing systems cannot incorporate incidents, weather or peak-hour variation.",
        "Many congestion displays colour by hard-coded threshold rules and do not state how a label was derived or what data it was computed from.",
    ]))
    bodyp(
        "A further limitation relevant to any data-driven replacement is that no "
        "self-serve traffic API publishes vehicle throughput, so any system that "
        "pretends to know traffic volume is fabricating it. The proposed system "
        "stores 'unknown' where an upstream source does not publish a value."
    )

    # --- Proposed System --------------------------------------------------------------
    H1("Proposed System")
    bodyp(
        "The proposed system is a modular, end-to-end pipeline with five "
        "standalone parts:"
    )
    story.extend(bullets([
        "Data ingestion: a provider interface with a labelled simulated provider and a live provider (TomTom Flow Segment Data + Open-Meteo weather), stored with source provenance on every row.",
        "Data pipeline and model: a six-stage, leakage-controlled preprocessing pipeline feeding a Random Forest classifier (v4.0.0) whose artifacts carry machine-readable provenance.",
        "Prediction engine: after a reading is stored it is scored against that segment's history and the forecast is persisted; insufficient history is an explicit error, never an invented prediction.",
        "Dashboard API + frontend: five aggregation endpoints (current, trends, locations, status, model) consumed by a static dashboard with a Leaflet map and Chart.js analytics.",
        "Hardened operation: a production configuration guard, one error envelope, upstream-failure classification, and full deployment documentation.",
    ]))
    bodyp(
        "Key design decisions: the ML layer never imports the web layer; routes stay "
        "thin while logic lives in services and all SQL in repositories; provenance "
        "is a first-class column; an unavailable capability is an explicit error "
        "(e.g. 422 insufficient_history, 503 model_not_available, 503 "
        "provider_not_configured); an unmeasured value is stored NULL, never "
        "estimated."
    )

    # --- System Architecture ---------------------------------------------------------
    H1("System Architecture")
    bodyp(
        "The architecture separates the browser, the FastAPI service layer, the "
        "independent ML pipeline, the PostgreSQL store and the external data "
        "providers. The background collection scheduler is opt-in and never blocks "
        "the request path."
    )
    story.append(architecture_diagram())
    story.append(P("Figure 1: System architecture (data providers -> collector -> FastAPI -> dashboard; model scored offline by the independent ml/ pipeline).", S["Caption"]))

    # --- Technologies Used ----------------------------------------------------------
    H1("Technologies Used")
    story.append(make_table(stack_rows, [75, 235, W - 310]))
    story.append(Spacer(1, 8))
    # --- Data Collection ---------------------------------------------------------
    H1("Data Collection")
    bodyp(
        "Observations are collected through a single provider interface "
        "(app/services/providers.py) so the storage, prediction and dashboard "
        "layers never depend on a vendor:"
    )
    story.extend(bullets([
        "SimulatedTrafficProvider - deterministic, seeded synthetic development data. Every record is stamped source='simulation' and every report, response and dashboard states it is not real traffic.",
        "RealTrafficProvider - live TomTom Flow Segment Data per monitored road: current and free-flow speed, confidence and road closure. A missing traffic API key fails with provider_not_configured rather than falling back to synthetic data.",
        "Weather - Open-Meteo forecast API alongside traffic (no key required in the free tier).",
        "Monitored roads - config/monitored_locations.json maps each coordinate to one of the six ML road segments (SEG-01..06); unknown segments are rejected on load.",
    ]))
    bodyp(
        "Two data facts are deliberately honest: vehicle_count is NULL on live "
        "readings because no self-serve traffic API publishes vehicle throughput, "
        "and there is no historical traffic feed - TomTom reports current conditions "
        "only - so history is accumulated by the scheduler or backfilled "
        "(POST /api/traffic/collect/history)."
    )

    # --- Data Preprocessing -----------------------------------------------------------
    H1("Data Preprocessing")
    bodyp("The six-stage offline pipeline (ml/preprocessing.py) runs independently of the API:")
    story.append(make_table([
        ("Stage", "Function"),
        ("Load", "Reads CSV or Parquet, records SHA-256, returns a copy"),
        ("Validate", "Schema, dtypes, missing values, duplicates, timestamp parseability, physical ranges"),
        ("Clean", "Resolves duplicate keys, nulls impossible values, imputes per segment; refuses to drop more than 20% of rows"),
        ("Features", "Calendar fields, speed_ratio, strictly backward lags and rolling means, one-hot / frequency encoding"),
        ("Target", "Preserves a supplied label, otherwise derives four congestion bands from speed_ratio"),
        ("Output", "Writes traffic_processed.csv plus a .meta.json sidecar and data_summary.md"),
    ], [70, W - 70]))
    story.append(Spacer(1, 8))
    bodyp(
        "Required dataset columns: timestamp, road_segment_id, avg_speed_kph, plus "
        "free_flow_speed_kph or speed_ratio. Optional when present: "
        "flow_veh_per_hr, occupancy_pct, temperature_c, precipitation_mm, "
        "weather_condition, is_incident."
    )

    # --- Feature Engineering ----------------------------------------------------------
    H1("Feature Engineering")
    bodyp("The feature set (26 features in model v4.0.0) is built to be leakage-safe:")
    story.extend(bullets([
        "Calendar features (hour, weekday and similar) to capture peak-hour variation.",
        "Static segment capacity and encoded segment and weather identity.",
        "Strictly backward lag and rolling-mean features over the same road segment - rolling windows are shifted so the current reading is never included.",
        "All imputation is per segment, so no segment inherits another's values.",
        "Contemporaneous measurements (avg_speed_kph, flow_veh_per_hr, occupancy_pct, speed_ratio) are excluded because they determine the label; supplying them to training or to the API is rejected.",
    ]))
    bodyp(
        "The congestion target is derived from speed_ratio = avg_speed_kph / "
        "free_flow_speed_kph using left-closed bands: free_flow [0.75, oo), moderate "
        "[0.40, 0.75), heavy [0.15, 0.40), severe (-oo, 0.15). This is a documented "
        "convention, not measured ground truth."
    )

    # --- Machine Learning Model ---------------------------------------------------------
    H1("Machine Learning Model")
    bodyp(
        "The model is a RandomForestClassifier, version 4.0.0 (26 features, four "
        "classes: free_flow, moderate, heavy, severe, label mapping 0-3). It is "
        "trained on a chronological 80/20 split - never a random one, because "
        "adjacent traffic observations are strongly autocorrelated - and evaluated "
        "with macro F1 as the headline metric so the rare 'severe' class is not "
        "ignored."
    )
    story.append(make_table([
        ("Property", "Value"),
        ("Estimator", "RandomForestClassifier"),
        ("Model version", "4.0.0"),
        ("Features", "26"),
        ("Classes", "free_flow | moderate | heavy | severe"),
        ("min_history_per_segment", "6"),
        ("Training data", "Simulated/development fixture (dataset_is_simulated=true)"),
        ("Validated on real traffic", "null / not performed"),
        ("Evaluation", "macro F1 as primary metric, against majority-class baseline"),
    ], [160, W - 160]))
    story.append(Spacer(1, 8))
    bodyp(
        "Training writes reproducible artifacts: traffic_model.pkl, preprocessor.pkl "
        "(feature order and label map - removing the train/serve skew failure mode), "
        "model_metadata.json (hyperparameters, split boundary, class balance, "
        "provenance), feature_importance.json, evaluation_results.json, "
        "model_report.md and confusion_matrix.png."
    )

    # --- Prediction Workflow ------------------------------------------------------------
    H1("Prediction Workflow")
    bodyp(
        "A forecast always reads a segment's stored history; a caller cannot "
        "manufacture the lag features a prediction depends on:"
    )
    story.extend(bullets([
        "POST /api/traffic/predict stores the reading first, scores it against the stored prior observations for the same road_segment_id, persists the forecast, and returns both.",
        "The model needs at least min_history_per_segment (6) prior readings; otherwise the request answers 422 insufficient_history naming how many readings exist and how many are required - never a fabricated number.",
        "Collection runs the same path automatically: POST /api/traffic/collect stores each reading first, then scores it and reports one outcome per reading (predicted, insufficient_history, model_unavailable, unscoreable_location, failed). A prediction failure can never cost an observation.",
        "history/latest: GET /api/traffic/predictions and /predictions/latest return the stored forecasts joined to the observations they describe.",
        "The dashboard model card reports dataset_is_simulated and validated_on_real_traffic next to any forecast it renders.",
    ]))

    # --- Real-Time Traffic Pipeline -----------------------------------------------------
    H1("Real-Time Traffic Pipeline")
    bodyp(
        "Live collection runs on an opt-in background scheduler "
        "(COLLECTION_ENABLED=true) or on a manual POST. Each cycle fetches current "
        "conditions per monitored location from TomTom Flow Segment Data and weather "
        "from Open-Meteo, normalises vendor fields onto the model's exact schema, "
        "stores with provenance, and auto-scores each new reading."
    )
    story.extend(bullets([
        "Upstream throttle: TRAFFIC_MIN_REQUEST_INTERVAL and TRAFFIC_MAX_LOCATIONS keep one cycle well inside TomTom's documented quota.",
        "Failure classification: 429 -> 503 provider_rate_limited; other 4xx/5xx -> 502; timeout -> 504; an unusable body -> 502 provider_invalid_response. Vendor bodies are never echoed (they contain the request URL and the API key).",
        "A failed collection cycle is counted in /api/dashboard/status, logged, and never fatal.",
        "Freshness: measured from stored observation timestamps against DATA_FRESHNESS_THRESHOLD_SECONDS and reported fresh / stale / unavailable.",
        "Data provenance: every stored row carries a source label (simulation, tomtom, or an API caller's value) and the collector status endpoint reports whether the active provider is serving synthetic data.",
    ]))

    # --- Weather Integration -------------------------------------------------------------
    H1("Weather Integration")
    bodyp(
        "The model requires weather, and no traffic vendor publishes it, so Open-Meteo "
        "is used: a keyless forecast API with a documented free allowance. The client "
        "requests only the fields the model consumes - temperature_2m, precipitation "
        "and weather_code - and maps WMO codes onto the model's three condition "
        "labels (clear / cloudy / rain); an out-of-vocabulary code logs a warning and "
        "resolves to the nearest safe label rather than failing a collection. "
        "timezone=GMT keeps stored times comparable with the UTC timestamps on "
        "observations (no half-day shift for a Chennai deployment). WEATHER_ENABLED "
        "can switch the request off, with the caveat that readings stored without "
        "weather cannot be scored."
    )
    bodyp(
        "Verification: a live one-off call to the Open-Meteo forecast endpoint "
        "returned HTTP 200 with current.temperature_2m, current.precipitation and "
        "current.weather_code present - exactly the fields the client reads."
    )

    # --- Backend API ---------------------------------------------------------------------
    H1("Backend API")
    bodyp("FastAPI exposes a documented REST API (OpenAPI at /docs) with one error envelope {error: {code, message, details}}:")
    story.append(make_table(api_rows, [62, 200, W - 262]))
    story.append(Spacer(1, 8))
    bodyp(
        "Health vs readiness: GET /api/health is a liveness probe; "
        "GET /api/dashboard/status reports readiness of each dependency (database, "
        "provider, scheduler, model, freshness). Dashboard reads are single-poll "
        "aggregations with a fixed query count (window functions, never N+1)."
    )

    # --- Frontend Dashboard ---------------------------------------------------------------
    H1("Frontend Dashboard")
    bodyp(
        "frontend/ ships a landing page and a dashboard (dashboard.html) with a "
        "sidebar (Overview, Live Traffic, Predictions, Analytics, Locations, System "
        "Status) and a location selector:"
    )
    story.extend(bullets([
        "Overview - six KPI cards, a Leaflet map with one marker per monitored location coloured by the predicted congestion class, and a per-location detail card.",
        "Live Traffic - current reading, forecast and freshness per location from /api/dashboard/current.",
        "Predictions - latest stored forecast and history.",
        "Analytics - Chart.js line chart (speed, free-flow and predicted class per point) for 1/6/12/24-hour windows from /api/dashboard/trends, plus the model card.",
        "Locations - readiness cards (e.g. '5 of 6 readings', stale/fresh, provider).",
        "System Status - database, providers, scheduler, model and freshness.",
        "Auto-refresh 15/30/60 s or manual, paused when the tab is hidden; every panel has loading, error and honest empty states; missing values render as N/A, never a placeholder number.",
    ]))
    bodyp(
        "Leaflet and Chart.js are vendored locally (no CDN), and every API-derived "
        "string is HTML-escaped (UI.esc) before insertion - pinned by contract tests."
    )

    # --- Database Design -----------------------------------------------------------------
    H1("Database Design")
    story.append(make_table(db_rows, [120, 150, W - 270]))
    story.append(Spacer(1, 8))
    bodyp(
        "Indexes cover the dashboard read paths: timestamp/location/road-name/segment "
        "on observations, and (observation_id, prediction_timestamp) plus "
        "observation/created on predictions. The schema is managed by a three-revision "
        "Alembic chain (table creation, nullable vehicle_count, history index). The "
        "DSN is resolved from application settings at runtime so no credential is "
        "ever committed to alembic.ini, and Alembic refuses to run without a target "
        "database rather than guessing one."
    )

    # --- Testing ---------------------------------------------------------------------------
    H1("Testing")
    bodyp(
        "The suite runs with no external services: database tests use an isolated "
        "in-memory SQLite database, provider tests drive httpx.MockTransport, and the "
        "migrations are verified by rendering their SQL for PostgreSQL offline."
    )
    story.append(make_table(test_rows, [200, W - 200]))
    story.append(Spacer(1, 8))
    bodyp("Coverage spans: the health contract, error envelope, configuration (including the production guard), DB wiring, schemas, provider response mappings, locations file, background scheduler, the full observation -> predict -> persist flow against the real trained artifact, auto-prediction failure modes, every dashboard endpoint with freshness behaviour, Alembic migrations, the ML pipeline, and the Stage 7/8 frontend contract and XSS/escaping tests.")

    # --- Results ----------------------------------------------------------------------------
    H1("Results")
    story.append(make_table(results_rows, [230, W - 230]))
    story.append(Spacer(1, 8))
    bodyp(
        "Verification detail: the full suite passed 450/450 on the final run "
        "(2 benign sklearn warnings). The Alembic round-trip on a fresh database "
        "reached head c3d4e5f6a7b8 with 2 tables, 12 indexes and the unique "
        "(location_id, timestamp) constraint, then downgraded and re-applied "
        "cleanly. Open-Meteo returned HTTP 200 live. TomTom live verification was "
        "not possible without a TRAFFIC_API_KEY and is covered by mock-transport "
        "contract tests instead."
    )

    # --- Benefits -----------------------------------------------------------------------------
    H1("Benefits")
    story.extend(bullets([
        "Forward-looking, not just descriptive: forecasts answer the question that can change driver behaviour.",
        "One tested pipeline from any data source: provider abstraction means the storage, prediction and dashboard layers never change when a vendor does.",
        "Honest by construction: simulated data is always labelled, missing values stay missing, and the model's limitations are displayed beside its forecasts.",
        "Leakage-safe machine learning: chronological splits, backward-only features and refused leak columns make reported metrics trustworthy at design level.",
        "Operationally safe: a production configuration guard, one error envelope, classified upstream failures, and an opt-in scheduler.",
        "Verifiable: 450 automated tests, compile checks, and documented live-endpoint verification where credentials permitted.",
    ]))

    # --- Limitations ----------------------------------------------------------------------------
    H1("Limitations")
    story.extend(bullets([
        "The model was trained on a simulated/development fixture; forecasts are demonstrative until the model is retrained and validated against real-world ground truth.",
        "vehicle_count is NULL on live readings because no self-serve traffic API publishes throughput.",
        "No historical traffic feed exists; a fresh location needs six prior readings before the first forecast.",
        "Live PostgreSQL round-trip was not executed in the development environment (no server); the migration chain was verified on SQLite and offline PostgreSQL DDL.",
        "Live TomTom verification requires a TRAFFIC_API_KEY, which was unavailable; the path is covered by mocked contract tests.",
        "The API is unauthenticated by design; gateway authentication and rate limiting are required before public exposure.",
    ]))

    # --- Future Enhancements ---------------------------------------------------------------------
    H1("Future Enhancements")
    story.extend(bullets([
        "Retrain and revalidate on a real urban traffic dataset once sufficient history is collected (the pipeline is ready and tested).",
        "Container image for the API and a CI pipeline running the test suite.",
        "Gateway authentication and rate limiting for public exposure.",
        "A historical or throughput data source should one become self-serve.",
        "A staging environment, plus a live PostgreSQL round-trip in CI.",
    ]))

    # --- Deployment ------------------------------------------------------------------------------
    H1("Deployment")
    bodyp(
        "Deployment is environment-driven and guard-railed. Copy .env.example to "
        ".env, set DATABASE_URL, apply the schema with alembic upgrade head, then "
        "start from the project root so the ml package is importable:"
    )
    story.append(CODE(".\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000", S["Code"]))
    bodyp(
        "When ENVIRONMENT=production the application refuses to start unless "
        "TRAFFIC_PROVIDER=real, TRAFFIC_API_KEY and a PostgreSQL DATABASE_URL are "
        "all set - a missing piece fails at boot naming the setting, never echoing "
        "the secret, so simulated data can never be served in production. "
        "DB_FAIL_FAST=true makes an unreachable database fail the same way. "
        "Liveness (/api/health) and readiness (/api/dashboard/status) are distinct, "
        "and guidance covers quota maths, one-worker operation, reverse-proxy and "
        "static-frontend serving and a troubleshooting table. Full detail: "
        "docs/deployment.md."
    )

    # --- Conclusion -------------------------------------------------------------------------------
    H1("Conclusion")
    bodyp(
        "The project delivers a complete, modular AI-based urban traffic congestion "
        "prediction system: a leakage-safe data pipeline, an evaluated Random Forest "
        "classifier, real-time ingestion behind a provider interface, an automatic "
        "prediction engine, a dashboard API and interactive frontend, and a hardened "
        "production configuration. It is verified by 450 automated tests and "
        "documented end to end."
    )
    bodyp(
        "The decisive caveat is stated rather than hidden: the training data is a "
        "simulated fixture, so current ML forecasts are demonstrative. The same, "
        "tested pipeline is ready to be retrained and validated against real-world "
        "ground truth as soon as sufficient real traffic observations are "
        "collected. Until then, this is a working, honest pipeline - not a "
        "field-validated traffic predictor."
    )

    # --- Appendix -----------------------------------------------------------------------------------
    H1("Appendix")
    H2("A. Commands used and verified")
    story.append(CODE("python -m pip install -r requirements.txt\n"
                      "Copy-Item .env.example .env   # then set DATABASE_URL, provider\n"
                      "alembic upgrade head          # apply schema; --sql to review offline\n"
                      ".\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend\n"
                      "pytest -q                     # 450 passed\n"
                      "python -m compileall -q backend ml migrations tests\n"
                      "node --check frontend/js/<file>        (repeat for every file)\n"
                      "python -m http.server 5500 --directory frontend   # serve the dashboard", S["Code"]))
    H2("B. Migration verification detail")
    story.append(make_table(migration_rows, [170, 190, W - 360]))
    story.append(Spacer(1, 8))
    H2("C. Repository layout (summary)")
    story.append(CODE("backend/app/   FastAPI application (api, schemas, services, repositories, database, core)\n"
                      "ml/            ML package (preprocessing, train, evaluate, predict, data, models)\n"
                      "frontend/      static dashboard (html, css, js, vendor/Leaflet + Chart.js)\n"
                      "migrations/    Alembic environment + 3 versions\n"
                      "tests/         pytest suite\n"
                      "docs/          architecture, api, ml-pipeline, roadmap, deployment, report, presentation", S["Code"]))
    H2("D. Honest-use statement")
    bodyp(
        "All metrics in this report describe the simulated development fixture. The "
        "model has not been validated on real-world ground truth; any presentation "
        "of a forecast as a field-tested number would be false."
    )


def build():
    story = []
    cover(story, S)
    body(story, S)

    doc = BaseDocTemplate(
        PDF_PATH, pagesize=A4, title="AI-Based Urban Traffic Congestion Prediction System - Project Report",
        author="Traffic Congestion Prediction Project",
        subject="College / internship project report",
        leftMargin=56, rightMargin=56, topMargin=52, bottomMargin=48,
    )

    def footer(canv, doc_inst):
        canv.saveState()
        canv.setFont("Calibri", 8)
        canv.setStrokeColor(GRID)
        canv.setLineWidth(0.5)
        canv.line(56, 40, A4[0] - 56, 40)
        canv.setFillColor(GREY)
        canv.drawString(56, 30, "AI-Based Urban Traffic Congestion Prediction System - Project Report")
        canv.drawRightString(A4[0] - 56, 30, "Page %d" % doc_inst.page)
        canv.restoreState()

    cover_frame = Frame(56, 48, A4[0] - 112, A4[1] - 100, id="cover")
    body_frame = Frame(56, 56, A4[0] - 112, A4[1] - 112, id="body")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=lambda c, d: None),
        PageTemplate(id="body", frames=[body_frame], onPage=footer),
    ])
    doc.build(story)
    return PDF_PATH


if __name__ == "__main__":
    out = build()
    print("WROTE", out)