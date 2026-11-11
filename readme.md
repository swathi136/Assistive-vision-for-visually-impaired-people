# 👁️ Vision Intelligence Dashboard

Real-time object detection, image captioning, OCR, and voice feedback — powered by YOLOv8, BLIP, and EasyOCR.

---

## ⚡ Quick VS Code Setup

### 1 — Prerequisites
- Python 3.10 or 3.11 (recommended). Download from https://python.org
- Git (optional, for version control)
- VS Code → https://code.visualstudio.com

### 2 — VS Code Extensions to install
Open VS Code → Extensions panel (`Ctrl+Shift+X`) → search and install:
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Black Formatter** (Microsoft) — optional, for auto-formatting

### 3 — Create a virtual environment
Open the VS Code terminal (`Ctrl+`` `):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4 — Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads ~2 GB of model weights (YOLOv8 + BLIP). They are cached after that.

### 5 — Select interpreter in VS Code
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose `./venv/Scripts/python.exe` (or `./venv/bin/python` on Mac/Linux).

### 6 — Run the app

**Option A — VS Code Run & Debug:**
Press `F5` (uses `.vscode/launch.json`). Browser opens at http://localhost:8501.

**Option B — Terminal:**
```bash
streamlit run app.py
```

### Deploy to Render

The repository includes a `render.yaml` Blueprint for deploying this Streamlit app.

1. Sign in to [Render](https://dashboard.render.com) and connect your GitHub account.
2. Select **New → Blueprint**, then choose this repository and the `main` branch.
3. Review the `vision-intelligence-dashboard` web service and apply the Blueprint.

The service runs Streamlit on Render's assigned port. The Blueprint selects the
`1c-2g` plan (1 CPU, 2 GB RAM) because the AI models need more memory than the
free plan provides. Render charges for paid compute according to your account's
current pricing.

---

## 🖥️ Modes

| Mode | Description |
|------|-------------|
| 📷 Live Webcam | Uses your default webcam (device 0). |
| 🎬 Video Demo | Upload an .mp4 / .mov / .avi file and process it. |
| 🖼️ Image Upload | Upload a single image for full analysis (+ optional OCR). |

---

## 🔧 Architecture

```
Display thread (Streamlit main)    ←  ~30 fps, shows raw or annotated frame
       │ puts frame every 3rd        
       ↓                             
  frame_q (maxsize=2, newest wins)  
       │                             
       ↓                             
Background worker thread           ←  YOLO + BLIP (blocks here, not on UI)
       │ puts result                 
       ↓                             
  result_q (maxsize=1, newest wins) 
       │                             
       ↑ non-blocking get           
Display thread updates UI widgets  
```

- **Frame skipping**: every 3rd frame is sent to ML; the rest display smoothly.
- **Queue draining**: the worker always processes the *newest* frame, discarding older queued frames.
- **Voice throttle**: TTS fires at most once every 5 seconds.

---

## 🐞 Common issues

| Problem | Fix |
|---------|-----|
| Webcam not opening | Make sure no other app is using it; try `cv2.VideoCapture(1)` if you have multiple cameras. |
| CUDA out of memory | Set `device = "cpu"` at the top of `app.py`. |
| `easyocr` slow first run | It downloads models; subsequent runs are fast. |
| Audio not playing | Browser may block autoplay; click anywhere on the page first. |