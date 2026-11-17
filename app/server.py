"""LeafScope local server: a hand-built HTML/CSS/JS interface served by Python's standard library.

    python app/server.py            ->  http://127.0.0.1:8000

Endpoints
    GET  /api/status        which examiners (trained models) are loaded
    GET  /api/samples       real PlantVillage test leaves for the "specimen drawer"
    GET  /api/sample/<n>    one of those images
    POST /api/diagnose      raw image bytes in the body -> diagnosis JSON
    GET  /api/trials        outputs/comparison.json (from evaluate.py)
    GET  /figures/<file>    figures written by train.py / evaluate.py
"""
import argparse
import base64
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config            # noqa: E402
import knowledge         # noqa: E402
import severity          # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 12 * 1024 * 1024
DISPLAY_SIDE = 448

PREDICTORS = {}
LOAD_NOTES = []
LOCK = threading.Lock()
SAMPLES = []


def load_models():
    try:
        import inference                      # needs PyTorch
    except Exception as e:                    # noqa: BLE001
        LOAD_NOTES.append("PyTorch is not installed here, so only the OpenCV lesion measurement is running.")
        return
    for name in config.MODEL_NAMES:
        if (config.CKPT_DIR / f"{name}.pt").exists():
            try:
                PREDICTORS[name] = inference.Predictor(name)
                print(f"loaded {name} on {PREDICTORS[name].device}")
            except Exception as e:            # noqa: BLE001
                LOAD_NOTES.append(f"{name}: {e}")
        else:
            LOAD_NOTES.append(f"{name}: no checkpoint (run train.py --model {name})")


def load_samples(n=60):
    if not config.SPLIT_FILE.exists():
        return
    sp = json.loads(config.SPLIT_FILE.read_text())
    rng = random.Random(7)
    items = list(sp["test"])
    rng.shuffle(items)
    # one sample per class first, so the drawer shows variety
    seen, picked = set(), []
    for rel, label in items:
        if label not in seen:
            seen.add(label)
            picked.append((rel, label))
    for it in items:
        if len(picked) >= n:
            break
        if it not in picked:
            picked.append(tuple(it))
    for rel, label in picked[:n]:
        SAMPLES.append({"path": str(Path(sp["root"]) / rel), "class_name": sp["class_names"][label], "rel": rel})


def data_url(img_bgr, ext=".jpg", quality=88):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img_bgr, params)
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def describe_top(top, class_names=None):
    out = []
    for t in top:
        info = knowledge.describe(t["class_name"])
        out.append({"class_name": t["class_name"], "prob": round(t["prob"], 4), **info})
    return out


def diagnose(raw: bytes, which: str):
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("That file could not be read as an image.")
    h, w = bgr.shape[:2]
    scale = DISPLAY_SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    sev = severity.estimate_severity(bgr)
    result = {
        "image": data_url(bgr),
        "size": [w, h],
        "severity": {
            "percent": sev["percent"], "grade": sev["grade"], "label": sev["label"],
            "leaf_pct_of_frame": sev["leaf_fraction_of_frame"],
            "overlay": data_url(severity.render_overlay(bgr, sev)),
        },
        "examiners": {},
        "notes": list(LOAD_NOTES),
    }
    if sev["leaf_fraction_of_frame"] < 12:
        result["notes"].append("The leaf fills very little of the frame; move closer for a reliable reading.")

    names = [n for n in PREDICTORS if which in ("both", n)]
    if names:
        import gradcam                       # imported lazily: it needs PyTorch
    for name in names:
        pred = PREDICTORS[name]
        with LOCK:
            r = pred.predict(rgb, topk=3, with_cam=True)
        cam = r["cam"]
        result["examiners"][name] = {
            "label": config.MODEL_LABELS[name],
            "top": describe_top(r["top"]),
            "ms": round(r["ms"], 1),
            "cam": data_url(gradcam.overlay(bgr, cam)),
            "attention": severity.attention_split(cam, sev["leaf_mask"], sev["lesion_mask"]),
        }
    if len(names) >= 2:
        tops = [result["examiners"][n]["top"][0]["class_name"] for n in names]
        result["agree"] = len(set(tops)) == 1
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "LeafScope/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[leafscope] " + fmt % args + "\n")

    def _send(self, code, body: bytes, ctype="application/json", cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _file(self, path: Path, cache="no-store"):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".png": "image/png",
                 ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}.get(
                     path.suffix.lower(), "application/octet-stream")
        self._send(200, path.read_bytes(), ctype, cache)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if p in ("/style.css", "/app.js"):
            return self._file(STATIC / p.lstrip("/"))
        if p == "/api/status":
            return self._json({
                "examiners": [{"name": n, "label": config.MODEL_LABELS[n], "device": str(pr.device),
                               "test": pr.meta["test"]} for n, pr in PREDICTORS.items()],
                "notes": LOAD_NOTES, "samples": len(SAMPLES)})
        if p == "/api/samples":
            n = int(parse_qs(u.query).get("n", ["12"])[0])
            return self._json([{"id": i, "accession": f"PV-{i + 1:03d}"} for i in range(min(n, len(SAMPLES)))])
        if p.startswith("/api/sample/"):
            try:
                i = int(p.rsplit("/", 1)[1])
                s = SAMPLES[i]
            except (ValueError, IndexError):
                return self._json({"error": "no such sample"}, 404)
            body = Path(s["path"]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Truth", base64.b64encode(json.dumps(
                {"class_name": s["class_name"], **knowledge.describe(s["class_name"])}).encode()).decode())
            self.send_header("Access-Control-Expose-Headers", "X-Truth")
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)
            return
        if p == "/api/trials":
            if config.COMPARISON_JSON.exists():
                return self._send(200, config.COMPARISON_JSON.read_bytes())
            return self._json({"models": {}})
        if p.startswith("/figures/"):
            name = Path(p).name                       # no directory traversal
            return self._file(config.FIG_DIR / name, cache="max-age=60")
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/diagnose":
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD:
            return self._json({"error": "Send one image under 12 MB."}, 413)
        raw = self.rfile.read(length)
        which = parse_qs(u.query).get("model", ["both"])[0]
        try:
            return self._json(diagnose(raw, which))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:                        # noqa: BLE001
            self.log_message("diagnose failed: %r", e)
            return self._json({"error": "Diagnosis failed on the server."}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    load_models()
    load_samples()
    for note in LOAD_NOTES:
        print("note:", note)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"LeafScope running at http://{a.host}:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
