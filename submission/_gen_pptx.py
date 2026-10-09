"""Generate the 13-slide project presentation (16:9) with python-pptx."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

OUT = r"C:\Users\RANJITH T\Desktop\TV\submission\AI_Traffic_Congestion_Prediction_Presentation.pptx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
STEEL = RGBColor(0x2E, 0x74, 0xB5)
LIGHT = RGBColor(0xDC, 0xE6, 0xF1)
PALE = RGBColor(0xF2, 0xF6, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x59, 0x59, 0x59)
AMBER = RGBColor(0x8A, 0x6D, 0x1D)
GREEN = RGBColor(0x54, 0x82, 0x35)
RED = RGBColor(0xC0, 0x00, 0x00)
SKY = RGBColor(0x9D, 0xC3, 0xE6)

FONT = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

SW, SH = 13.333, 7.5


def slide():
    return prs.slides.add_slide(BLANK)


def rect(s, x, y, w, h, fill, line=None, round_=False, radius=0.06):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    if round_:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    return shp


def arrow(s, x, y, w, h, direction, fill=STEEL):
    kind = {"down": MSO_SHAPE.DOWN_ARROW, "right": MSO_SHAPE.RIGHT_ARROW}.get(direction)
    shp = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def tbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tf


def para(tf, text, size=14, color=NAVY, bold=False, italic=False, align=PP_ALIGN.LEFT,
         space_before=0, space_after=4, first=False, line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    f = r.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    if line:
        p.line_spacing = line
    return p


def bullets(tf, items, size=14, color=NAVY, start=0):
    first = True
    for level, text in items:
        if level == 0:
            t = "\u2022  " + text
        else:
            t = "      \u2013  " + text
        para(tf, t, size=size - (4 if level else 0), color=color,
             align=PP_ALIGN.LEFT, first=first, space_after=7)
        first = False


def header(s, kicker, title, page):
    rect(s, 0, 0, SW, 1.35, NAVY)
    rect(s, 0, 1.35, SW, 0.045, STEEL)
    tf = tbox(s, 0.6, 0.12, SW - 1.2, 0.32)
    para(tf, kicker.upper(), size=11, color=SKY, bold=True, first=True)
    tf = tbox(s, 0.6, 0.42, SW - 1.2, 0.85)
    para(tf, title, size=29, color=WHITE, bold=True, first=True)
    footer(s, page)


def footer(s, page):
    line = rect(s, 0.6, 7.14, SW - 1.2, 0.014, LIGHT)
    tf = tbox(s, 0.6, 7.18, SW - 1.2, 0.25)
    tf2 = tbox(s, 0.6, 7.18, 9.0, 0.25)
    para(tf2, "AI-Based Urban Traffic Congestion Prediction System", size=10, color=GREY, first=True)
    tf3 = tbox(s, 9.4, 7.18, 3.3, 0.25)
    para(tf3, "Page %d" % page, size=10, color=GREY, align=PP_ALIGN.RIGHT, first=True)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def breakdown_box(s, x, y, w, h, title, lines, fill=PALE, tcolor=NAVY, radius=0.1):
    box = rect(s, x, y, w, h, fill, line=STEEL, round_=True, radius=radius)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.14)
    tf.margin_right = Inches(0.14)
    tf.margin_top = Inches(0.08)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.bold = True; r.font.size = Pt(13); r.font.name = FONT; r.font.color.rgb = tcolor
    for ln in lines:
        p = tf.add_paragraph(); p.space_before = Pt(3)
        r = p.add_run(); r.text = "\u2013  " + ln
        r.font.size = Pt(11); r.font.name = FONT; r.font.color.rgb = GREY
    return box


# ---------------------------------------------------------------------------
# Slide 1 - Title
# ---------------------------------------------------------------------------
s = slide()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 4.55, SW, 0.05, STEEL)
tf = tbox(s, 0.9, 1.7, SW - 1.8, 0.4, anchor=MSO_ANCHOR.BOTTOM)
para(tf, "PROJECT REPORT \u2013 AI-BASED URBAN TRAFFIC CONGESTION PREDICTION SYSTEM",
     size=15, color=SKY, bold=True, align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, 0.9, 2.25, SW - 1.8, 1.7)
para(tf, "AI-Based Urban Traffic\nCongestion Prediction System",
     size=46, color=WHITE, bold=True, align=PP_ALIGN.CENTER, first=True, line=1.05)
tf = tbox(s, 1.8, 4.75, SW - 3.6, 0.9)
para(tf, "A machine-learning pipeline that predicts per-road-segment congestion "
         "ahead of time and visualises it on a live dashboard.",
     size=16, color=WHITE, align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, 0.9, 5.55, SW - 1.8, 0.4)
para(tf, "Collect  \u2192  Predict  \u2192  Visualise", size=14, color=SKY,
     align=PP_ALIGN.CENTER, first=True)
step = 1.0
for i, t in enumerate(["Stages 1\u20138 complete", "451 automated tests passing",
                       "Report + deck included"]):
    rect(s, 0.9 + i * (step + 2.9), 6.25, 3.0, 0.62, STEEL, round_=True, radius=0.25)
    tf = tbox(s, 0.9 + i * (step + 2.9), 6.25, 3.0, 0.62, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, t, size=13, color=WHITE, bold=True, align=PP_ALIGN.CENTER, first=True)
notes(s, "Title. Framing: reactive today, forward-looking with ML. Demo only: "
         "training data is simulated and labelled as such in every artifact.")

# ---------------------------------------------------------------------------
# Slide 2 - Problem Statement
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 02", "Problem Statement", 2)
tf = tbox(s, 0.9, 1.9, SW - 1.8, 4.6)
bullets(tf, [
    (0, "Congestion is discovered too late \u2013 commuters learn about it when "
        "they are already inside it."),
    (0, "Reactive traffic control answers \u201cwhere is it congested now?\u201d, "
        "not \u201cwill this road be congested in 30 minutes?\u201d."),
    (0, "Static timetables cannot capture incidents, weather and peak-hour variation."),
    (0, "The question that changes behaviour is forward-looking and per road segment."),
    (0, "The gaps a data-driven system must fill:"),
    (1, "no vendor publishes traffic throughput, so vehicle_count is stored as unknown"),
    (1, "any system that pretends to know real traffic volume is fabricating it"),
])
box = rect(s, 0.9, 5.75, SW - 1.8, 1.0, LIGHT, round_=True)
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.2)
para(tf, "Need: a forward-looking per-segment congestion estimate that is honest "
         "about what it can and cannot know.", size=15, color=NAVY, bold=True,
     align=PP_ALIGN.CENTER, first=True)
notes(s, "Problem: reactivity. Static timetables miss incidents, weather, peaks. "
         "Key honesty point: no vendor publishes throughput.")

# ---------------------------------------------------------------------------
# Slide 3 - Objectives
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 03", "Objectives and Approach", 3)
tf = tbox(s, 0.9, 1.85, SW - 1.8, 4.7)
bullets(tf, [
    (0, "Collect  \u2013  real urban traffic observations into PostgreSQL, with "
        "provenance on every row."),
    (0, "Engineer  \u2013  leakage-safe features from time, road geometry and weather."),
    (0, "Train  \u2013  a Random Forest classifier and score it against an honest "
        "majority-class baseline."),
    (0, "Serve  \u2013  documented REST API returning one error envelope."),
    (0, "Show  \u2013  current and predicted congestion on a map and charts."),
    (0, "Guard  \u2013  boot-time production guard so a misconfigured deployment fails "
        "loudly instead of silently serving simulation."),
])
box = rect(s, 0.9, 5.8, SW - 1.8, 0.9, PALE, round_=True)
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.2)
para(tf, "Every result traceable to data; nothing fabricated.",
     size=16, color=STEEL, bold=True, align=PP_ALIGN.CENTER, first=True)
notes(s, "Objectives in five steps plus the production guard. Nothing fabricated.")

# ---------------------------------------------------------------------------
# Slide 4 - Proposed Solution
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 04", "Proposed Solution", 4)
parts = [
    ("Data ingestion", ["provider interface", "labelled simulation OR live TomTom + Open-Meteo"]),
    ("Pipeline + model", ["6-stage, leakage-controlled", "Random Forest v4.0.0, 26 features", "artifacts carry provenance"]),
    ("Prediction engine", ["scored against segment history", "insufficient history = explicit error"]),
    ("Dashboard API + frontend", ["5 aggregation endpoints", "Leaflet map + Chart.js analytics"]),
    ("Production guard", ["no silent simulation", "fails at boot naming the missing setting"]),
]
pos = [(0.9, 1.9), (4.9, 1.9), (8.9, 1.9), (0.9, 4.35), (4.9, 4.35)]
for (x, y), (t, ls) in zip(pos, parts):
    breakdown_box(s, x, y, 3.55, 2.2, t, ls)
box = rect(s, 8.9, 4.35, 3.55, 2.2, LIGHT, round_=True, radius=0.08)
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.16); tf.margin_right = Inches(0.16)
para(tf, "One interface, two providers, one honest pipeline", size=14, color=NAVY,
     bold=True, align=PP_ALIGN.CENTER, first=True)
p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Nothing downstream builds on an unproven assumption."
r.font.size = Pt(12); r.font.name = FONT; r.font.color.rgb = GREY
notes(s, "Five component parts. Emphasise provider interface and honest pipeline.")

# ---------------------------------------------------------------------------
# Slide 5 - System Architecture
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 05", "System Architecture", 5)
# Browser / frontend
breakdown_box(s, 5.0, 1.75, 3.35, 0.85, "Browser / Frontend",
              ["static HTML5 + CSS3 + JS", "Leaflet map \u00b7 Chart.js \u00b7 auto-refresh"], radius=0.12)
arrow(s, 6.36, 2.62, 0.6, 0.4, "down")
# FastAPI
breakdown_box(s, 4.8, 3.08, 3.7, 1.0, "FastAPI backend",
              ["api/ \u00b7 schemas/ \u00b7 services/ \u00b7 repositories/", "one error envelope \u00b7 CORS"], radius=0.1)
# Data providers (left)
breakdown_box(s, 0.9, 3.08, 3.0, 1.0, "Data providers",
              ["Simulated (labelled)", "Real: TomTom", "Weather: Open-Meteo"])
arrow(s, 3.92, 3.42, 0.85, 0.26, "right")
tf = tbox(s, 0.9, 4.16, 3.0, 0.3)
para(tf, "opt-in scheduler:  collect \u2192 predict", size=11, color=GREY, italic=True, first=True)
# Down to ML + PostgreSQL row
arrow(s, 3.42, 4.42, 0.6, 0.35, "down")
arrow(s, 9.32, 4.42, 0.6, 0.35, "down")
breakdown_box(s, 1.9, 4.85, 4.2, 1.2, "ML pipeline  (ml/)",
              ["Random Forest v4.0.0 \u00b7 26 features", "offline train / evaluate / predict",
               "independent of the web layer"], radius=0.1)
breakdown_box(s, 7.25, 4.85, 4.2, 1.2, "PostgreSQL 14+",
              ["traffic_observations", "traffic_predictions", "Alembic migrations"], radius=0.1)
notes(s, "Browser -> FastAPI -> PostgreSQL. ML package independent of the web layer; "
         "providers behind one interface; opt-in scheduler.")

# ---------------------------------------------------------------------------
# Slide 6 - Data & ML Pipeline
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 06", "Data & ML Pipeline", 6)
tf = tbox(s, 0.9, 1.9, SW - 1.8, 2.1)
para(tf, "Six-stage offline pipeline", size=15, color=STEEL, bold=True, first=True)
stages = ["Load", "Validate", "Clean", "Featurise", "Target", "Write"]
for i, st in enumerate(stages):
    rect(s, 0.9 + i * 2.0, 2.6, 1.85, 0.75, NAVY if i == 3 else STEEL, round_=True, radius=0.15)
    tf = tbox(s, 0.9 + i * 2.0, 2.6, 1.85, 0.75, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, st, size=14, color=WHITE, bold=True, align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, 0.9, 3.75, SW - 1.8, 2.9)
bullets(tf, [
    (0, "Leakage controls: contemporaneous measurements refused at training AND at "
        "the API boundary; backward-only lags; chronological split."),
    (0, "Random Forest \u2013 strong on tabular data, little tuning, feature importances "
        "explain predictions."),
    (0, "4 congestion levels (free flow / moderate / heavy / severe)."),
    (0, "Macro F1 is the headline metric \u2013 accuracy optimisers ignore rare severe "
        "congestion."),
    (0, "Every artifact stores its own feature order + provenance \u2013 no train/serve "
        "feature-skew failure mode, and the fixture is labelled simulated."),
])
notes(s, "6-stage pipeline. Leakage controls. Random Forest rationale. Macro F1. "
         "Provenance removes feature skew. Simulated fixture labelled.")

# ---------------------------------------------------------------------------
# Slide 7 - Real-Time Traffic Collection
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 07", "Real-Time Traffic Collection", 7)
cols = [
    ("Provider interface", [
        "SimulatedTrafficProvider \u2013 labelled, always",
        "RealTrafficProvider \u2013 TomTom Flow Segment Data",
        "WeatherProvider \u2013 Open-Meteo",
        "one code path for scheduled + manual POST /collect"]),
    ("Collection behaviour", [
        "throttled per cycle",
        "opt-in via COLLECTION_ENABLED=true",
        "timestamped with source provenance on every row",
        "stored first, predicted second (auto-prediction)"]),
    ("Honesty + failure mode", [
        "missing provider key fails loudly, never silently simulated",
        "vehicle_count unknown \u2192 stored as NULL (no vendor publishes throughput)",
        "production requires TRAFFIC_PROVIDER=real + TRAFFIC_API_KEY",
        "no self-serve throughput API \u2013 a system claiming volume fabricates it"]),
]
for i, (t, ls) in enumerate(cols):
    breakdown_box(s, 0.9 + i * 3.9, 1.9, 3.7, 3.5, t, ls)
tf = tbox(s, 0.9, 5.75, SW - 1.8, 0.9)
para(tf, "Live outside the suite: Open-Meteo call verified (HTTP 200). "
         "TomTom needs a real key.", size=14, color=AMBER, italic=True, first=True)
notes(s, "Providers behind one interface; throttle; opt-in; provenance. Node that "
         "missing key fails loudly; NULL for vehicle_count; production guard.")

# ---------------------------------------------------------------------------
# Slide 8 - Weather Integration
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 08", "Weather Integration", 8)
# provider traces
rect(s, 0.9, 1.95, 5.0, 0.7, STEEL, round_=True, radius=0.18)
tf = tbox(s, 0.9, 1.95, 5.0, 0.7, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Open-Meteo  \u2013  no API key required", size=15, color=WHITE,
     bold=True, align=PP_ALIGN.CENTER, first=True)
arrow(s, 5.92, 2.17, 0.6, 0.26, "right")
rect(s, 6.54, 1.95, 5.9, 0.7, NAVY, round_=True, radius=0.18)
tf = tbox(s, 6.54, 1.95, 5.9, 0.7, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "appended to each observation before featurisation", size=15, color=WHITE,
     bold=True, align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, 0.9, 3.0, SW - 1.8, 3.4)
bullets(tf, [
    (0, "Live call verified in this project: HTTP 200, fields current.temperature_2m, "
        "current.precipitation, current.weather_code."),
    (0, "Weather cycles share the throttled provider pipeline \u2013 one cycle, "
        "weather + traffic together."),
    (0, "Weather fields become model features alongside time and road geometry."),
    (0, "In production the weather provider keeps working with TRAFFIC_PROVIDER=real."),
    (0, "Weather is standard practice for congestion models \u2013 rain and temperature "
        "shift speed distributions."),
])
box = rect(s, 0.9, 5.8, SW - 1.8, 0.9, LIGHT, round_=True)
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.2)
para(tf, "No key, no fabrications \u2013 the weather source was exercised live.",
     size=16, color=NAVY, bold=True, align=PP_ALIGN.CENTER, first=True)
notes(s, "Open-Meteo live-verified (HTTP 200). Fields temperature_2m, precipitation, "
         "weather_code. Throttled with traffic; weather is a model feature.")

# ---------------------------------------------------------------------------
# Slide 9 - Dashboard
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 09", "Dashboard", 9)
tf = tbox(s, 0.9, 1.9, SW - 1.8, 4.4)
bullets(tf, [
    (0, "Leaflet map coloured by predicted congestion class (free / moderate / heavy / severe)."),
    (0, "Chart.js trend chart \u2013 speed vs forecast over 1 / 6 / 12 / 24 hours."),
    (0, "Auto-refresh; live traffic tab \u2192 map \u2192 analytics \u2192 system status."),
    (0, "Five aggregation endpoints: current, trends, locations, status, model "
        "\u2013 fixed queries, no N+1."),
    (0, "Honest empty states \u2013 \u201cN/A\u201d is never a fabricated zero; "
        "uncollected roads report null and unavailable."),
    (0, "Model provenance shown next to every forecast (dataset_is_simulated: true), "
        "and every API-derived string is HTML-escaped before it reaches the page."),
])
notes(s, "Leaflet map + Chart.js. Honest empty states. Provenance next to forecasts. "
         "XSS: all API-derived strings escaped.")

# ---------------------------------------------------------------------------
# Slide 10 - Testing & Results
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 10", "Testing & Results", 10)
rect(s, 0.9, 1.9, 6.4, 4.3, PALE, line=STEEL, round_=True, radius=0.05)
tf = tbox(s, 1.15, 2.05, 5.9, 3.0)
para(tf, "Verification \u2013 451 automated tests", size=16, color=NAVY, bold=True, first=True)
bullets(tf, [
    (0, "observe \u2192 predict \u2192 persist against the real artifact"),
    (0, "every dashboard endpoint, freshness boundaries, fixed query counts"),
    (0, "Alembic: apply \u2192 verify \u2192 reverse \u2192 re-apply + offline PostgreSQL DDL"),
    (0, "provider contracts via mocked transports; security + secret-leak tests"),
    (0, "Stage 8: production-guard + XSS tests"),
    (0, "no external services required to run the suite"),
], size=13)
rect(s, 7.65, 1.9, 4.8, 4.3, LIGHT, line=STEEL, round_=True, radius=0.05)
tf = tbox(s, 7.95, 2.15, 4.2, 3.9)
para(tf, "On the simulated fixture", size=16, color=NAVY, bold=True, first=True)
para(tf, "macro F1   0.737", size=24, color=GREEN, bold=True, space_before=12)
para(tf, "accuracy   0.856", size=24, color=GREEN, bold=True)
para(tf, "vs majority-class baseline", size=13, color=GREY, space_before=10)
para(tf, "macro F1 0.214 \u00b7 accuracy 0.750", size=18, color=GREY, bold=True)
para(tf, "These numbers show the pipeline runs \u2013 they are not a statement about "
         "real traffic.", size=12, color=AMBER, italic=True, space_before=14)
notes(s, "451 tests, no external services. Results are demo-only: macro F1 0.737, "
         "accuracy 0.856 vs baseline 0.214/0.750. Not real-traffic claims.")

# ---------------------------------------------------------------------------
# Slide 11 - Potential Benefits & Drawbacks
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 11", "Potential Benefits & Drawbacks", 11)
breakdown_box(s, 0.9, 1.9, 5.75, 4.35, "Benefits", [
    "forward-looking \u2013 reroute before congestion, not inside it",
    "less idling \u2192 saved travel time, fuel and emissions",
    "reproducible, inspectable model, not arbitrary threshold rules",
    "one provider interface \u2013 retraining-ready on real data",
    "cheap to pilot on free-tier sources (TomTom, keyless Open-Meteo)",
])
breakdown_box(s, 6.85, 1.9, 5.6, 4.35, "Drawbacks", [
    "data gaps: current-conditions-only, quota-limited, no throughput",
    "six prior readings per location before any forecast is possible",
    "simulated-data model \u2013 demonstrative until revalidated",
    "privacy and liability around derived travel behaviour",
    "vendor dependence \u00b7 equity \u00b7 false-confidence risk",
])
box = rect(s, 0.9, 6.45, SW - 1.8, 0.55, LIGHT, round_=True)
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.2)
para(tf, "The honest drawbacks are designed in \u2013 labelling, provenance, production guard; "
         "the rest are named in the report.", size=13, color=NAVY, bold=True,
     align=PP_ALIGN.CENTER, first=True)
notes(s, "Discuss benefits then drawbacks. The honest ones are engineered in (labelling, "
         "provenance, production guard); the rest are stated in the report.")

# ---------------------------------------------------------------------------
# Slide 12 - Limitations / Future / Deployed
# ---------------------------------------------------------------------------
s = slide()
header(s, "Slide 12", "Limitations \u00b7 Future \u00b7 Deployed", 12)
breakdown_box(s, 0.9, 1.9, 3.85, 4.4, "Limitations", [
    "training data is a simulated fixture",
    "vehicle_count NULL on live readings",
    "no historical feed yet \u2013 scheduler accumulates it",
    "API unauthenticated until a gateway is added",
])
breakdown_box(s, 4.95, 1.9, 3.85, 4.4, "Future", [
    "retrain + revalidate on real ground truth",
    "gateway auth and rate limiting",
    "live TomTom + PostgreSQL round-trip in CI",
    "staging environment",
])
breakdown_box(s, 9.0, 1.9, 3.45, 4.4, "Deployed now", [
    "Docker image + docker-compose",
    "CI: 451 tests + image build, green",
    "single-URL public demo (Render)",
    "live TomTom collection enabled",
])
notes(s, "Honest scope, then what is already shipped and running: Docker, CI, and the "
         "public single-URL demo.")

# ---------------------------------------------------------------------------
# Slide 13 - Conclusion
# ---------------------------------------------------------------------------
s = slide()
rect(s, 0, 0, SW, SH, NAVY)
tf = tbox(s, 0.9, 1.3, SW - 1.8, 0.5)
para(tf, "SLIDE 13 \u00b7 CONCLUSION", size=13, color=SKY, bold=True, align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, 0.9, 1.8, SW - 1.8, 0.9)
para(tf, "A working, honest end-to-end pipeline", size=34, color=WHITE, bold=True,
     align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, 1.6, 2.9, SW - 3.2, 2.6)
bullets(tf, [
    (0, "Collect \u2192 predict \u2192 visualise, verified by 451 automated tests."),
    (0, "Leakage-safe machine learning with provenance on every artifact."),
    (0, "Demonstrated \u2013 not field-validated: forecasts improve once real ground "
        "truth flows through the same tested pipeline."),
    (0, "Containerised, CI-green and deployed at a public single URL \u2013 ready for "
        "retraining on real data and gateway hardening."),
], size=17, color=WHITE)
tf = tbox(s, 0.9, 5.85, SW - 1.8, 1.0)
para(tf, "Questions.", size=30, color=SKY, bold=True, align=PP_ALIGN.CENTER, first=True)
tf = tbox(s, 0.9, 6.7, SW - 1.8, 0.4)
para(tf, "Project report, deployment guide and full API docs shipped alongside the deck.",
     size=12, color=WHITE, align=PP_ALIGN.CENTER, first=True)
notes(s, "Closing. Honest scope + live deployment + next steps. Questions.")

prs.save(OUT)
print("WROTE", OUT)