import io
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "florix_ai_best.keras"
LABELS_PATH = BASE_DIR / "models" / "florix_class_names.json"
IMAGE_SIZE = (160, 160)

st.set_page_config(page_title="FLORAGUARD AI | Plant Health Intelligence", page_icon="🌿", layout="wide")

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--green:#087f50;--green2:#13a16a;--ink:#17251e;--muted:#6b7d73;--line:#dfeae3;--soft:#f3f8f4;--white:#fff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:linear-gradient(180deg,#f8fbf8 0%,#f1f7f2 100%)}
.block-container{max-width:1400px;padding-top:1.4rem;padding-bottom:2.2rem}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.hero{background:linear-gradient(120deg,#073e2b 0%,#087f50 55%,#31a66d 100%);padding:30px 34px;border-radius:24px;color:white;box-shadow:0 12px 30px #0b57351c;margin-bottom:22px;position:relative;overflow:hidden}
.hero:after{content:'✳';position:absolute;right:30px;top:-40px;font-size:190px;color:#ffffff12}
.brand{font:800 13px 'Manrope',sans-serif;letter-spacing:2.4px;text-transform:uppercase;color:#c6f4d8}
.hero h1{font:800 clamp(28px,3.4vw,43px) 'Manrope',sans-serif;color:white;line-height:1.12;margin:10px 0 8px}
.hero p{color:#e0f3e7;margin:0;max-width:760px;font-size:15px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}
.chip{border:1px solid #ffffff50;background:#ffffff12;border-radius:30px;padding:7px 11px;color:#f4fff7;font-size:12px;font-weight:600}
.panel{background:white;border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:0 5px 18px #163b2410;margin-bottom:16px}
.panel-title{font:800 18px 'Manrope',sans-serif;color:#193b2b;margin:0 0 4px}
.eyebrow{font-size:10px;font-weight:800;letter-spacing:1.6px;color:var(--green);text-transform:uppercase;margin-bottom:8px}
.muted{color:var(--muted);font-size:13px}
.metric-card{background:var(--soft);border:1px solid var(--line);border-radius:15px;padding:14px 16px;height:100%}
.metric-label{font-size:11px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted);font-weight:700}
.metric-value{font:800 23px 'Manrope',sans-serif;color:var(--green);margin-top:5px;overflow-wrap:anywhere}
.badge-good{display:inline-block;background:#e0f6e8;color:#087344;border-radius:20px;padding:5px 10px;font-size:12px;font-weight:800}
.badge-warn{display:inline-block;background:#fff0d8;color:#89510b;border-radius:20px;padding:5px 10px;font-size:12px;font-weight:800}
.result-name{font:800 25px 'Manrope',sans-serif;color:#0a7549;overflow-wrap:anywhere}
.result-note{background:#f4f9f5;border-left:4px solid #13a16a;border-radius:8px;padding:12px 14px;color:#385747;font-size:13px;margin-top:14px}
.feature-card{background:#f7faf7;border:1px solid var(--line);border-radius:14px;padding:15px;margin:8px 0}
.feature-card h4{font:800 14px 'Manrope',sans-serif;margin:0 0 7px;color:#204a34}
.feature-card p{font-size:13px;color:#5b6f62;margin:0}
.stButton>button{border-radius:12px!important;font-weight:800!important;min-height:44px;border:1px solid #087f50!important}
.stButton>button[kind="primary"]{background:linear-gradient(100deg,#087f50,#13a16a)!important;color:white!important}
[data-testid="stFileUploader"]{background:#f7faf7;border:1px dashed #b9d8c3;border-radius:15px;padding:12px}
[data-testid="stTabs"] button{font-weight:700}
hr{border-color:var(--line)}
.small-foot{color:#829188;font-size:11px;text-align:center;padding:18px 0 5px;line-height:1.7}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="brand">🌿 Floraguard AI · Plant Health Intelligence</div>
  <h1>Every leaf tells a story.<br>Let's understand yours.</h1>
  <p>AI-assisted plant disease prediction, image quality validation, severity estimation and practical plant-care guidance in one workspace.</p>
  <div class="chips"><span class="chip">✳ Smart Image Validation</span><span class="chip">◉ Severity Estimation</span><span class="chip">❋ Uncertainty Analysis</span><span class="chip">🔬 Explainable AI</span><span class="chip">▤ PDF Reports</span></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        # Support either index->label or label->index JSON layouts.
        try:
            return [labels[str(i)] for i in range(len(labels))]
        except Exception:
            return [k for k, _ in sorted(labels.items(), key=lambda item: item[1])]
    return list(labels)

def pretty_label(label):
    return re.sub(r"\s+", " ", str(label).replace("___", " — ").replace("_", " ").replace(",", " ")).strip()

def validate_leaf_image(image):
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings = []
    if min(w, h) < 128: warnings.append("The image resolution is small; a larger image may improve reliability.")
    if brightness < 45: warnings.append("The image looks dark. Try taking the photo in natural light.")
    elif brightness > 220: warnings.append("The image may be overexposed. Reduce glare and retake the photo.")
    if sharpness < 35: warnings.append("The image may be blurry. Keep the leaf in focus.")
    quality = "Good" if not warnings else ("Needs improvement" if len(warnings) > 1 else "Fair")
    return {"quality":quality,"warnings":warnings,"metrics":{"width":w,"height":h,"brightness":round(brightness,1),"sharpness":round(sharpness,1)}}

def estimate_severity(image):
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    mask = ((sat > 35) & (val > 25) & (((hue >= 15) & (hue <= 100)) | ((hue >= 5) & (hue < 15) & (sat > 70))))
    if int(mask.sum()) < 100:
        return {"severity":"Unavailable","affected_pct":None,"note":"Leaf/background separation was too uncertain for a useful estimate."}
    lesion = (((hue < 15) | (hue > 100)) & (sat > 45) & (val > 25)) & mask
    pct = float(lesion.sum() / mask.sum() * 100)
    level = "Low" if pct < 8 else "Moderate" if pct < 20 else "High"
    return {"severity":level,"affected_pct":round(pct,1),"note":"A rough color-based visual estimate, not a validated agronomic severity measurement."}

def uncertainty_analysis(probs):
    p = np.asarray(probs, dtype=np.float64)
    sorted_p = np.sort(p)[::-1]
    top = float(sorted_p[0]); second = float(sorted_p[1]) if len(sorted_p)>1 else 0.0
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(max(len(p),2)))
    return {"top":top,"second":second,"margin":top-second,"entropy":entropy}

def care_guidance(label):
    low = str(label).lower()
    if "healthy" in low:
        return "Healthy-looking leaf", ["Continue crop-appropriate watering and nutrition.","Monitor new growth and check both sides of leaves regularly.","Maintain airflow and clean gardening tools."]
    if any(x in low for x in ["bacterial", "bact"]):
        return "Possible bacterial condition", ["Avoid overhead watering where practical.","Remove severely affected material with clean tools if appropriate for the crop.","Avoid handling plants while wet and seek local agricultural guidance before applying treatments."]
    if any(x in low for x in ["virus", "mosaic", "curl"]):
        return "Possible viral condition", ["Inspect nearby plants for similar symptoms.","Clean tools and manage insect vectors according to local guidance.","Ask an agricultural extension professional about crop-specific control options."]
    if any(x in low for x in ["fung", "rust", "mildew", "scorch", "spot", "rot", "blight", "scab", "mold"]):
        return "Possible fungal or leaf-spot condition", ["Avoid prolonged leaf wetness and improve airflow where possible.","Remove badly affected leaves when appropriate and dispose of them safely.","Confirm the disease and crop before choosing any treatment."]
    return "Plant-health follow-up", ["Compare the result with symptoms across several leaves.","Check watering, light, drainage and recent weather conditions.","Confirm the diagnosis with a local agricultural expert before applying treatments."]

def gradcam_overlay(model, image, class_index):
    """Best-effort Grad-CAM for the EfficientNetB0 model saved by this project."""
    try:
        base = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()), None)
        if base is None:
            base = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
        if base is None:
            return None
        conv_layer = next((layer for layer in reversed(base.layers) if isinstance(layer, tf.keras.layers.Conv2D)), None)
        if conv_layer is None:
            return None
        conv_index = base.layers.index(conv_layer)
        base_index = model.layers.index(base)
        conv_probe = tf.keras.Model(base.input, conv_layer.output)
        arr = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)[None, ...]
        # Continue the forward pass from the selected convolution layer so the
        # GradientTape can measure the class score with respect to its activations.
        with tf.GradientTape() as tape:
            conv_out = conv_probe(arr, training=False)
            z = conv_out
            for layer in base.layers[conv_index + 1:]:
                try: z = layer(z, training=False)
                except TypeError: z = layer(z)
            for layer in model.layers[base_index + 1:]:
                try: z = layer(z, training=False)
                except TypeError: z = layer(z)
            score = z[:, class_index]
        grads = tape.gradient(score, conv_out)
        if grads is None: return None
        weights = tf.reduce_mean(grads, axis=(1,2))[0]
        fmap = conv_out[0]
        heat = tf.reduce_sum(fmap * weights, axis=-1)
        heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
        heat = cv2.resize(heat.numpy(), image.size)
        heat_u8 = np.uint8(255 * heat)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        base_img = np.asarray(image.convert("RGB"))
        return Image.fromarray(np.uint8(np.clip(0.58 * base_img + 0.42 * colored, 0, 255)))
    except Exception:
        return None

def make_pdf(image, label, probs, labels, quality, severity, uncertainty, care_title, care_items, heatmap):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, Table, TableStyle
        from reportlab.lib.utils import ImageReader
    except ImportError as e:
        raise RuntimeError("PDF support needs reportlab. Add reportlab to requirements.txt and redeploy.") from e
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.7*cm, leftMargin=1.7*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Brand", parent=styles["Title"], textColor=colors.HexColor("#087f50"), alignment=TA_CENTER, fontSize=22, leading=27))
    story = [Paragraph("FLORAGUARD AI", styles["Brand"]), Paragraph("Plant Health Intelligence · Analysis Report", styles["Heading2"]), Spacer(1, 8), Paragraph("Generated: " + datetime.now().strftime("%d %b %Y, %H:%M"), styles["Normal"]), Spacer(1, 12)]
    img_buf = io.BytesIO(); image.convert("RGB").thumbnail((700,500)); image.convert("RGB").save(img_buf, format="JPEG"); img_buf.seek(0)
    story.append(PDFImage(img_buf, width=10*cm, height=7*cm, kind="proportional")); story.append(Spacer(1, 10))
    story.append(Paragraph("Predicted class: " + pretty_label(label), styles["Heading2"]))
    story.append(Paragraph(f"Model confidence: {float(np.max(probs))*100:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Image quality: {quality['quality']} · Resolution: {quality['metrics']['width']} × {quality['metrics']['height']} px", styles["Normal"]))
    story.append(Paragraph(f"Color-based affected-area estimate: {severity['severity']}" + (f" ({severity['affected_pct']}%)" if severity['affected_pct'] is not None else ""), styles["Normal"]))
    story.append(Spacer(1, 8)); story.append(Paragraph("Top predictions", styles["Heading3"]))
    data = [["Class", "Score"]] + [[pretty_label(labels[i]), f"{float(probs[i])*100:.2f}%"] for i in np.argsort(probs)[-3:][::-1]]
    table = Table(data, colWidths=[12*cm, 3*cm]); table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#087f50")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#dfeae3")),("PADDING",(0,0),(-1,-1),7)])); story.append(table)
    story.append(Spacer(1, 10)); story.append(Paragraph(care_title, styles["Heading3"]))
    for item in care_items: story.append(Paragraph("• " + item, styles["Normal"]))
    if heatmap is not None:
        story.append(Spacer(1, 10)); story.append(Paragraph("Model attention visualization (Grad-CAM)", styles["Heading3"]))
        hb=io.BytesIO(); heatmap.save(hb,format="PNG"); hb.seek(0); story.append(PDFImage(hb,width=10*cm,height=7*cm,kind="proportional"))
    story += [Spacer(1, 14), Paragraph("Important: This AI output is informational, may be incorrect, and is not a confirmed agricultural diagnosis. Confirm crop-specific treatment with a qualified local agricultural professional.", styles["Italic"])]
    doc.build(story); return buffer.getvalue()

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Model files are missing. Expected models/florix_ai_best.keras and models/florix_class_names.json.")
    st.stop()

try:
    model = load_model()
    class_names = load_labels()
    if len(class_names) != int(model.output_shape[-1]):
        st.error(f"Label mismatch: model outputs {model.output_shape[-1]} classes, but labels file contains {len(class_names)}.")
        st.stop()
except Exception as exc:
    st.error("The model or class labels could not be loaded.")
    st.exception(exc)
    st.stop()

left, right = st.columns([0.92, 1.28], gap="large")
with left:
    st.markdown('<div class="panel"><div class="eyebrow">STEP 01 · PLANT SCANNER</div><div class="panel-title">📸 Start your plant check</div><p class="muted">Upload a clear, well-lit leaf photo. For best results, keep the leaf in focus.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(image, caption="Uploaded leaf", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please try a JPG or PNG file.")
            image = None
    else:
        image = None
        st.markdown('<div class="panel"><div style="font-size:46px;text-align:center">🌱</div><div class="panel-title" style="text-align:center">Your leaf photo goes here</div><p class="muted" style="text-align:center">Supported formats: JPG, JPEG, PNG</p></div>', unsafe_allow_html=True)
    analyze = st.button("🌿 Analyze My Plant", type="primary", use_container_width=True, disabled=image is None)
    st.markdown('<div class="panel"><div class="panel-title">Tips for better results</div><div class="feature-card"><h4>✓ Natural lighting</h4><p>Avoid shadows, glare and very dark images.</p></div><div class="feature-card"><h4>✓ Focus on the leaf</h4><p>Keep the leaf sharp and visible in the frame.</p></div><div class="feature-card"><h4>✓ Keep the background simple</h4><p>Try to show the leaf clearly without clutter.</p></div></div>', unsafe_allow_html=True)

if analyze and image is not None:
    st.session_state["flora_last_image"] = image
    st.session_state["flora_result"] = None
    with st.spinner("Checking image quality and analyzing leaf..."):
        quality = validate_leaf_image(image)
        arr = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32)[None, ...]
        probs = model.predict(arr, verbose=0)[0]
        best = int(np.argmax(probs))
        severity = estimate_severity(image)
        uncertainty = uncertainty_analysis(probs)
        care_title, care_items = care_guidance(class_names[best])
        heatmap = gradcam_overlay(model, image, best)
        st.session_state["flora_result"] = {"image":image,"quality":quality,"probs":probs,"best":best,"severity":severity,"uncertainty":uncertainty,"care_title":care_title,"care_items":care_items,"heatmap":heatmap}

result = st.session_state.get("flora_result")
if result:
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="eyebrow">FEATURE 02 · DISEASE DETECTION</div><div class="panel-title">🧠 Plant health analysis</div></div>', unsafe_allow_html=True)
    q = result["quality"]; probs = result["probs"]; best = result["best"]; severity = result["severity"]; uncertainty = result["uncertainty"]
    r1, r2, r3, r4 = st.columns(4)
    with r1: st.markdown(f'<div class="metric-card"><div class="metric-label">Model confidence</div><div class="metric-value">{probs[best]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with r2: st.markdown(f'<div class="metric-card"><div class="metric-label">Image quality</div><div class="metric-value">{q["quality"]}</div></div>', unsafe_allow_html=True)
    with r3: st.markdown(f'<div class="metric-card"><div class="metric-label">Visual severity</div><div class="metric-value">{severity["severity"]}</div></div>', unsafe_allow_html=True)
    with r4: st.markdown(f'<div class="metric-card"><div class="metric-label">Prediction margin</div><div class="metric-value">{uncertainty["margin"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    main_col, side_col = st.columns([1.1, 0.9], gap="large")
    with main_col:
        st.markdown(f'<div class="panel"><div class="eyebrow">PREDICTION RESULT</div><div class="result-name">{pretty_label(class_names[best])}</div><div style="margin:12px 0"><span class="badge-good">{probs[best]*100:.2f}% model score</span></div><div class="result-note">Model confidence is not a guarantee of correctness. Use the image checks and symptoms as supporting information.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="eyebrow">TOP 3 PREDICTIONS</div><div class="panel-title">Most likely classes</div>', unsafe_allow_html=True)
        for i in np.argsort(probs)[-3:][::-1]:
            st.write(f"**{pretty_label(class_names[int(i)])}** · {float(probs[i])*100:.2f}%")
            st.progress(float(probs[i]))
        st.markdown('</div>', unsafe_allow_html=True)
    with side_col:
        st.markdown('<div class="panel"><div class="eyebrow">FEATURE 01 · IMAGE VALIDATION</div><div class="panel-title">📷 Photo quality check</div>', unsafe_allow_html=True)
        m=q["metrics"]
        st.write(f"**Resolution:** {m['width']} × {m['height']} px")
        st.write(f"**Brightness:** {m['brightness']}")
        st.write(f"**Sharpness score:** {m['sharpness']}")
        if q["warnings"]:
            for warning in q["warnings"]: st.warning(warning)
        else: st.success("The photo passed the basic image-quality checks.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="eyebrow">FEATURE 03 · DIAGNOSTICS</div><div class="panel-title">📊 Advanced diagnostics</div>', unsafe_allow_html=True)
        if severity["affected_pct"] is not None: st.write(f"**Estimated discolored area:** {severity['affected_pct']}%")
        st.write(f"**Normalized uncertainty:** {uncertainty['entropy']*100:.1f}%")
        st.write(f"**Top-two score gap:** {uncertainty['margin']*100:.2f} percentage points")
        st.caption(severity["note"])
        if uncertainty["margin"] < 0.15: st.warning("The top predictions are relatively close. Consider checking another clear leaf photo.")
        st.markdown('</div>', unsafe_allow_html=True)
    care_col, explain_col = st.columns(2, gap="large")
    with care_col:
        st.markdown(f'<div class="panel"><div class="eyebrow">FEATURE 04 · PLANT CARE</div><div class="panel-title">🌿 {result["care_title"]}</div></div>', unsafe_allow_html=True)
        for item in result["care_items"]: st.markdown(f"- {item}")
    with explain_col:
        st.markdown('<div class="panel"><div class="eyebrow">FEATURE 05 · EXPLAINABLE AI</div><div class="panel-title">🔬 Grad-CAM visualization</div></div>', unsafe_allow_html=True)
        if result["heatmap"] is not None:
            st.image(result["heatmap"], caption="Grad-CAM overlay (highlighted areas indicate model attention, not a verified lesion mask)", use_container_width=True)
        else:
            st.info("Grad-CAM could not be generated for this saved model graph. Prediction and diagnostics are available; the heatmap helper may need adjustment for this model version.")
    st.markdown('<div class="panel"><div class="eyebrow">FEATURE 06 · REPORT EXPORT</div><div class="panel-title">📄 Export your complete analysis</div><p class="muted">Download a PDF with the image, prediction scores, diagnostics and care guidance.</p></div>', unsafe_allow_html=True)
    try:
        pdf_bytes = make_pdf(result["image"], class_names[best], probs, class_names, q, severity, uncertainty, result["care_title"], result["care_items"], result["heatmap"])
        st.download_button("⬇️ Generate / Download PDF Report", data=pdf_bytes, file_name=f"floraguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", type="primary")
    except Exception as exc:
        st.error(f"Could not prepare PDF report: {exc}")
else:
    st.markdown('<div class="panel"><div class="eyebrow">YOUR WORKSPACE</div><div class="panel-title">Your plant health report will appear here</div><p class="muted">Upload a leaf image and choose <b>Analyze My Plant</b> to see predictions, image checks, diagnostics, care guidance, explainability and report export.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="small-foot"><b>FLORAGUARD AI · PLANT HEALTH INTELLIGENCE</b><br>AI-assisted results may be incorrect and are not confirmed agricultural diagnoses. Confirm crop-specific treatments with a qualified local agricultural professional.</div>', unsafe_allow_html=True)
