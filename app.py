import os
import html
import json
import tempfile
from pathlib import Path
from io import BytesIO
from datetime import datetime

import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageStat, ImageFilter

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "florix_ai_best.keras"
LABELS_PATH = BASE_DIR / "models" / "florix_class_names.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Class labels file not found: {LABELS_PATH}")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)
input_shape = model.input_shape
IMG_SIZE = int(input_shape[1]) if isinstance(input_shape, tuple) and input_shape[1] else 160
if len(class_names) != int(model.output_shape[-1]):
    raise ValueError(
        f"Class-label count ({len(class_names)}) does not match model output "
        f"({model.output_shape[-1]}). Check the model and label JSON files."
    )


# --- Original Colab notebook cell 11 ---
def validate_leaf_image(image):
    if image is None:
        return {"valid":False,"quality":"No image","message":"Upload a leaf image.","warnings":[],"metrics":{}}
    img = image.convert("RGB")
    w,h = img.size
    gray = ImageOps.grayscale(img)
    brightness = float(ImageStat.Stat(gray).mean[0])
    sharpness = float(gray.filter(ImageFilter.FIND_EDGES).getextrema()[1])
    warnings = []
    if min(w,h) < 100: warnings.append("Image resolution is small.")
    if brightness < 35: warnings.append("Image may be too dark.")
    if brightness > 225: warnings.append("Image may be overexposed.")
    if sharpness < 15: warnings.append("Image may be blurry.")
    valid = min(w,h) >= 100 and 20 <= brightness <= 245
    return {"valid":valid,"quality":"Good image quality" if valid and not warnings else "Image needs attention",
            "message":"Image checks passed." if valid else "Use a clearer, well-lit image.",
            "warnings":warnings,"metrics":{"width":w,"height":h,"brightness":round(brightness,2),"sharpness":round(sharpness,2)}}

def analyze_prediction_uncertainty(probabilities):
    p = np.clip(np.asarray(probabilities,dtype=float),1e-12,1)
    order = np.argsort(p)[::-1]
    top, second = float(p[order[0]]), float(p[order[1]])
    margin = top-second
    entropy = float(-np.sum(p*np.log(p))/np.log(len(p)))
    if top < .5 or margin < .15:
        status, recommendation = "Uncertain prediction","Retake the image and seek expert confirmation."
    elif top < .75:
        status, recommendation = "Moderate certainty","Compare alternatives and verify symptoms."
    else:
        status, recommendation = "Model favors one class","Treat this as screening, not a confirmed diagnosis."
    return {"status":status,"top_probability":round(top*100,2),"second_probability":round(second*100,2),
            "margin":round(margin*100,2),"normalized_entropy":round(entropy,4),"recommendation":recommendation}

def estimate_leaf_severity(image):
    arr = np.asarray(image.convert("RGB").resize((IMG_SIZE,IMG_SIZE)),dtype=np.float32)
    r,g,b = arr[...,0],arr[...,1],arr[...,2]
    green = (g > r*.85) & (g > b*.85) & (g > 35)
    discolored = (r > g*1.05) | (b > g*1.08) | (g < 65)
    affected = float(np.sum(discolored & green)/max(np.sum(green),1)*100)
    severity = "Low visible discoloration" if affected < 10 else ("Moderate visible discoloration" if affected < 30 else "High visible discoloration")
    return {"affected_percent":round(affected,2),"severity":severity,"message":"Rough color-based estimate only; lighting and background affect results."}

def get_care_recommendation(predicted_class):
    if "healthy" in predicted_class.lower():
        return {"description":"The model predicts a healthy-class image.","actions":["Continue appropriate watering and normal crop care.","Monitor leaves for new symptoms.","Keep tools clean."]}
    return {"description":"The model predicts a possible disease. Confirm before treatment.","actions":["Inspect several leaves for spreading symptoms.","Avoid unnecessary wetting of foliage and improve airflow where suitable.","Seek local agricultural guidance before applying pesticides."]}


# --- Original Colab notebook cell 13 ---
def generate_gradcam(image, model):
    image = image.convert("RGB").resize((IMG_SIZE,IMG_SIZE))
    x = np.asarray(image,dtype=np.float32)[None,...]
    base = next((layer for layer in model.layers if isinstance(layer,tf.keras.Model)),None)
    if base is None:
        raise ValueError("EfficientNet base model not found.")
    conv = base.get_layer("top_conv")
    extractor = tf.keras.Model(base.inputs,conv.output)
    tail, started = [], False
    for layer in base.layers:
        if layer.name == "top_conv":
            started = True
            continue
        if started: tail.append(layer)
    head, after_base = [], False
    for layer in model.layers:
        if layer is base:
            after_base = True
            continue
        if after_base: head.append(layer)
    with tf.GradientTape() as tape:
        features = extractor(x,training=False)
        tape.watch(features)
        z = features
        for layer in tail: z = layer(z,training=False)
        for layer in head: z = layer(z,training=False)
        idx = tf.argmax(z[0])
        score = z[:,idx]
    grads = tape.gradient(score,features)
    if grads is None:
        raise ValueError("Grad-CAM gradients unavailable for this architecture.")
    weights = tf.reduce_mean(grads,axis=(0,1,2))
    heat = tf.maximum(tf.reduce_sum(features[0]*weights,axis=-1),0)
    heat = heat/(tf.reduce_max(heat)+1e-8)
    heat = tf.image.resize(heat[...,None],(IMG_SIZE,IMG_SIZE)).numpy().squeeze()
    colored = (plt.get_cmap("jet")(heat)[...,:3]*255).astype(np.uint8)
    overlay = np.clip(.6*np.asarray(image).astype(np.float32)+.4*colored,0,255).astype(np.uint8)
    return image, heat, overlay


# --- Original Colab notebook cell 15 ---
def generate_flora_report(leaf_image,predicted_class,probabilities,class_names,severity_result=None,uncertainty_result=None,gradcam_image=None):
    path = str(Path(tempfile.gettempdir()) / f"FloraGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    doc = SimpleDocTemplate(path,pagesize=letter,rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="FloraTitle",parent=styles["Title"],textColor=colors.HexColor("#087f50")))
    story = [Paragraph("FloraGuard AI — Plant Health Report",styles["FloraTitle"]),
             Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}",styles["Normal"]),Spacer(1,12)]
    buf = BytesIO()
    leaf_image.convert("RGB").save(buf,format="JPEG"); buf.seek(0)
    story += [PDFImage(buf,width=2.2*inch,height=2.2*inch),Spacer(1,12),
              Paragraph(f"<b>Prediction:</b> {html.escape(str(predicted_class))}",styles["Normal"]),Spacer(1,8)]
    probs = np.asarray(probabilities); order = np.argsort(probs)[::-1][:3]
    data = [["Rank","Class","Score"]] + [[str(i+1),html.escape(str(class_names[int(j)])),f"{probs[j]*100:.2f}%"] for i,j in enumerate(order)]
    table = Table(data,colWidths=[.6*inch,4.5*inch,1*inch])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#087f50")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#dcebe1")),("PADDING",(0,0),(-1,-1),6)]))
    story += [table,Spacer(1,12)]
    if severity_result:
        story.append(Paragraph(f"<b>Visual estimate:</b> {html.escape(str(severity_result.get('severity')))} ({severity_result.get('affected_percent','N/A')}%)",styles["Normal"]))
        story.append(Paragraph(html.escape(str(severity_result.get("message",""))),styles["BodyText"]))
    if uncertainty_result:
        story.append(Paragraph(f"<b>Uncertainty:</b> {html.escape(str(uncertainty_result.get('status')))}",styles["Normal"]))
        story.append(Paragraph(html.escape(str(uncertainty_result.get("recommendation",""))),styles["BodyText"]))
    if gradcam_image is not None:
        try:
            ob = BytesIO()
            if isinstance(gradcam_image,Image.Image): gradcam_image.convert("RGB").save(ob,format="JPEG")
            else: Image.fromarray(np.asarray(gradcam_image).astype(np.uint8)).save(ob,format="JPEG")
            ob.seek(0); story += [Spacer(1,10),Paragraph("Grad-CAM overlay",styles["Heading2"]),PDFImage(ob,width=2.8*inch,height=2.8*inch)]
        except Exception as exc: print("Grad-CAM omitted from PDF:",exc)
    story += [Spacer(1,12),Paragraph("Disclaimer: This AI output is not a confirmed diagnosis. Confirm disease and treatment with an agricultural expert.",styles["BodyText"])]
    doc.build(story)
    return path


# --- Original Colab notebook cell 17 ---

FLORA_LAST_ANALYSIS = None

def flora_analyze_all(image):
    global FLORA_LAST_ANALYSIS
    if image is None:
        FLORA_LAST_ANALYSIS = None
        msg = "<div class='card'>Upload a leaf image to begin.</div>"
        return msg,msg,msg,msg,msg,None
    q = validate_leaf_image(image); m = q["metrics"]
    warnings_html = "".join("<li>"+html.escape(str(w))+"</li>" for w in q["warnings"])
    q_html = f"<div class='card'><div class='eyebrow'>IMAGE QUALITY</div><h3>{html.escape(q['quality'])}</h3><p>{html.escape(q['message'])}</p><p class='muted'>Resolution: {m['width']} × {m['height']} · Brightness: {m['brightness']} · Sharpness: {m['sharpness']}</p><ul>{warnings_html}</ul></div>"
    if not q["valid"]:
        FLORA_LAST_ANALYSIS = None
        msg = "<div class='card'>Please upload a clearer, well-lit leaf image.</div>"
        return q_html,msg,msg,msg,msg,None
    x = np.asarray(image.convert("RGB").resize((IMG_SIZE,IMG_SIZE)),dtype=np.float32)[None,...]
    probs = model.predict(x,verbose=0)[0]; top = np.argsort(probs)[::-1][:3]
    idx = int(top[0]); name = str(class_names[idx]); conf = float(probs[idx])*100
    unc = analyze_prediction_uncertainty(probs); sev = estimate_leaf_severity(image); care = get_care_recommendation(name)
    if conf < 50: status,color,bg,msg = "LOW CONFIDENCE","#b42318","#fff1f0","Retake the image and seek expert confirmation."
    elif conf < 75: status,color,bg,msg = "MODERATE CONFIDENCE","#946200","#fff8e6","Compare alternatives and verify visible symptoms."
    else: status,color,bg,msg = "HIGH MODEL CONFIDENCE","#087f50","#edf8f0","The prediction may still be incorrect; confirm symptoms."
    pred_html = f"<div class='card'><div class='eyebrow'>DISEASE PREDICTION</div><h2>{html.escape(name.replace('___',' — ').replace('_',' '))}</h2><div class='confidence-box'><div class='muted'>Model confidence</div><div class='confidence'>{conf:.2f}%</div><div class='track'><div class='fill' style='width:{conf:.2f}%'></div></div></div><div class='warning' style='color:{color};background:{bg};border-color:{color}55'><b>{status}</b><p>{msg}</p></div><p class='muted'>Confidence is not a guarantee of correctness.</p></div>"
    rows = ""
    for rank,j in enumerate(top,1):
        score=float(probs[j])*100
        rows += f"<div class='rank-row'><span class='rank'>{rank:02}</span><div class='rank-main'><div class='rank-label'><span>{html.escape(str(class_names[int(j)]).replace('___',' — ').replace('_',' '))}</span><b>{score:.2f}%</b></div><div class='track'><div class='fill' style='width:{score:.2f}%'></div></div></div></div>"
    top_html = f"<div class='card'><div class='eyebrow'>ALTERNATIVE PREDICTIONS</div><h3>Top 3 predictions</h3>{rows}</div>"
    diag_html = f"<div class='card'><div class='eyebrow'>ADVANCED DIAGNOSTICS</div><h3>Visible discoloration estimate</h3><div class='confidence'>{sev['affected_percent']}%</div><p>{html.escape(sev['severity'])}</p><p>{html.escape(sev['message'])}</p><hr><h3>Uncertainty</h3><b>{html.escape(unc['status'])}</b><p class='muted'>Top {unc['top_probability']}% · Second {unc['second_probability']}% · Margin {unc['margin']}%</p><p>{html.escape(unc['recommendation'])}</p></div>"
    care_html = f"<div class='card'><div class='eyebrow'>PLANT CARE</div><h3>Recommended next steps</h3><p>{html.escape(care['description'])}</p><ul>{''.join('<li>'+html.escape(str(a))+'</li>' for a in care['actions'])}</ul><p class='muted'>Confirm the disease before treatment.</p></div>"
    gradcam = None
    try:
        _,_,overlay = generate_gradcam(image,model)
        gradcam = Image.fromarray(overlay)
    except Exception as exc:
        print("Grad-CAM unavailable:",exc)
    FLORA_LAST_ANALYSIS = {"image":image.copy(),"predicted_class":name,"probabilities":np.asarray(probs).copy(),"severity":sev,"uncertainty":unc,"gradcam":gradcam}
    return q_html,pred_html,top_html,diag_html,care_html,gradcam

def download_flora_pdf(image):
    global FLORA_LAST_ANALYSIS
    if image is None: raise gr.Error("Upload a leaf image first.")
    FLORA_LAST_ANALYSIS = None
    flora_analyze_all(image)
    if FLORA_LAST_ANALYSIS is None: raise gr.Error("Analysis did not complete. Try a clearer image.")
    r = FLORA_LAST_ANALYSIS
    try:
        return generate_flora_report(r["image"],r["predicted_class"],r["probabilities"],class_names,r["severity"],r["uncertainty"],r["gradcam"])
    except Exception as exc:
        raise gr.Error(f"PDF generation failed: {exc}")

flora_css = """
.gradio-container{max-width:1400px!important;width:100%!important;margin:auto!important;padding:18px!important;background:#f4f8f5!important;font-family:Arial,sans-serif!important}
#hero{padding:26px 30px;margin-bottom:18px;border-radius:16px;background:linear-gradient(120deg,#075b3d,#13a16a);color:white}
#hero h1{color:white!important;font-size:clamp(26px,3vw,38px);margin:10px 0}
#hero p{color:#e5f6eb!important;line-height:1.6}
.card{box-sizing:border-box;width:100%;padding:16px;margin-bottom:12px;border:1px solid #dcebe1;border-radius:13px;background:white}
.card h2{color:#087f50;overflow-wrap:anywhere}.card h3{color:#173b2e;margin:7px 0 10px}
.eyebrow{color:#087f50;font-size:10px;font-weight:800;letter-spacing:1.3px;margin-bottom:8px}
.muted{font-size:12px;color:#6b7e75;line-height:1.6}
.confidence-box{background:#edf7f0;padding:14px;border-radius:10px;margin:12px 0}
.confidence{font-size:28px;font-weight:800;color:#087f50;margin:5px 0 10px}
.track{width:100%;height:7px;background:#e7eee9;border-radius:20px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#43c785,#087f50);border-radius:20px}
.warning{padding:12px;border:1px solid;border-radius:10px;margin:12px 0;font-size:12px;line-height:1.6}.warning p{margin:5px 0 0;color:#52675c}
.rank-row{display:flex;gap:10px;align-items:center;padding:12px 0;border-bottom:1px solid #edf2ee}
.rank{width:36px;height:36px;flex-shrink:0;display:flex;align-items:center;justify-content:center;border-radius:10px;background:#e8f6ed;color:#087f50;font-weight:800}
.rank-main{min-width:0;flex:1}.rank-label{display:flex;justify-content:space-between;gap:8px;font-size:12px;margin-bottom:7px;overflow-wrap:anywhere}.rank-label b{color:#087f50;white-space:nowrap}
#analyze-btn{background:linear-gradient(100deg,#087f50,#13a16a)!important;color:white!important}
#footer{padding:18px;text-align:center;color:#718278;font-size:12px;line-height:1.8}
@media(max-width:760px){.gradio-container{padding:10px!important}#hero{padding:22px 18px}}
"""

with gr.Blocks(title="FLORIX AI | Plant Health",theme=gr.themes.Soft(primary_hue="green",secondary_hue="emerald",neutral_hue="slate"),css=flora_css) as app:
    gr.HTML("<div id='hero'><h1>FLORIX AI 🌿</h1><p>AI-powered plant disease detection with confidence scores, visual explanations, and actionable plant-care recommendations.</p></div>")
    with gr.Row(equal_height=True):
        with gr.Column(scale=4,min_width=280):
            with gr.Group():
                gr.Markdown("### 📸 Plant Scanner")
                image_input=gr.Image(type="pil",label="Upload a clear leaf image",height=280)
                analyze_btn=gr.Button("🌿 Analyze My Plant",variant="primary",elem_id="analyze-btn")
                gr.Markdown("**Tips**\n- Use natural lighting.\n- Keep the leaf in focus.\n- Avoid blurry images.")
        with gr.Column(scale=6,min_width=320):
            gr.Markdown("### 🧠 Plant Health Analysis")
            quality_output=gr.HTML("<div class='card'>Image quality results appear here.</div>")
            prediction_output=gr.HTML("<div class='card'>Prediction results appear here.</div>")
            top_output=gr.HTML("<div class='card'>Top three predictions appear here.</div>")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1,min_width=280):
            gr.Markdown("### 📊 Advanced Diagnostics")
            diagnostics_output=gr.HTML("<div class='card'>Diagnostics appear here.</div>")
        with gr.Column(scale=1,min_width=280):
            gr.Markdown("### 🔬 Explainable AI")
            gradcam_output=gr.Image(label="Grad-CAM overlay",height=280)
    gr.Markdown("### 🌿 Plant-Care Recommendations")
    care_output=gr.HTML("<div class='card'>Recommendations appear here.</div>")
    gr.Markdown("### 📄 Export report")
    with gr.Row():
        pdf_btn=gr.Button("Generate PDF Report",variant="primary")
        pdf_output=gr.File(label="Download report",interactive=False)
    analyze_btn.click(fn=flora_analyze_all,inputs=[image_input],outputs=[quality_output,prediction_output,top_output,diagnostics_output,care_output,gradcam_output])
    pdf_btn.click(fn=download_flora_pdf,inputs=[image_input],outputs=[pdf_output])
    gr.HTML("<div id='footer'><b>FLORIX AI · PLANT HEALTH INTELLIGENCE</b><br>AI predictions and visual estimates are not confirmed diagnoses. Confirm treatment decisions with an agricultural expert.</div>")

app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
