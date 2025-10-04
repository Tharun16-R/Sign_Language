# Sign Language Detection and Translation (Flask + PyTorch)

This project provides a web-based AI system that:

- Detects sign language gestures from webcam in real time and converts them to text.
- Converts text to sign by playing per-letter sign videos (demo).
- Optional speech output using gTTS.

## Project Structure

```
sign-language-ai/
├─ app.py              # Flask backend for inference
├─ model/              # Model, dataset, utils, inference
├─ static/             # Frontend static files
├─ templates/          # HTML templates
├─ requirements.txt    # Dependencies
├─ capture_data.py     # Script to collect keypoints using webcam
├─ train_model.py      # Train sign detection model
└─ realtime_infer.py   # Real-time camera inference script
```

## Quickstart

1) Create and activate a Python 3.10+ environment.

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Run the Flask server (binds to 0.0.0.0:8080):

```
python app.py
```

Open http://localhost:8080 to view the basic frontend.

4) Local real-time inference (Python + OpenCV + MediaPipe):

```
python realtime_infer.py
```

5) Collect data to train your custom signs:

```
python capture_data.py --label HELLO --sequences 30 --frames 30
```
Data is saved under `data/processed/HELLO/` as `.npz` sequences.

6) Train the model:

```
python train_model.py --data_dir data/processed --epochs 20 --batch_size 32 --lr 1e-3
```
Weights are saved to `model/weights/best_model.pth`.

## API

- GET `/health` -> `{status: 'ok'}`
- POST `/predict` -> JSON `{ "sequence": [[...feature_1..feature_N], ...] }` returns `{ "prediction": "WORD", "probs": [..] }`
- POST `/text_to_speech` -> JSON `{ "text": "Hello" }` returns an mp3 file.
- GET `/text_to_sign?text=Hello` -> returns JSON mapping letters to demo video URLs.

## Frontend

A minimal page `templates/index.html` uses the webcam. The JS (`static/js/app.js`) captures hand landmarks in the browser using MediaPipe Hands and sends sliding-window sequences to `/predict` to show live predictions.

If MediaPipe JS is unavailable in your environment, you can still use `realtime_infer.py` which captures and infers via Python.

## Deployment

- The app binds to `0.0.0.0:8080` to run on Gradient or containers.
- You can proxy this backend behind a static frontend (Vercel/DO). Point your frontend to the backend `/predict` endpoint.

### Deploy on Gradient (Free Credits)

You can deploy this service using your Gradient credits as a containerized web app.

1) Ensure the repo contains the provided `Dockerfile` at the project root.

2) Push the repo to a Git host (GitHub/GitLab) or upload as a workspace.

3) In Gradient, create a Deployment (or Workflows/Services depending on UI version):
   - Source: your repo and branch.
   - Build: Dockerfile at `/Dockerfile`.
   - Ports: expose `8080`.
   - Command: `python app.py` (already configured as the container `CMD`).
   - Environment variables (optional):
     - `LABELS=HELLO,THANKS,ILOVEYOU`
     - `SEQ_LEN=30`
     - `MODEL_WEIGHTS=model/weights/best_model.pth`
   - Hardware: select a GPU instance if you plan to run PyTorch GPU (optional for inference on CPU).

4) Build and deploy. Once running, note the public URL. Test:

```
curl -s https://<your-gradient-url>/health
curl -s -X POST https://<your-gradient-url>/predict \
  -H 'Content-Type: application/json' \
  -d '{"sequence": [[0,0,0], [0,0,0]] }'
```

For a valid prediction, send a `(SEQ_LEN x 63)` array as described above.

5) Frontend integration: host a static site (e.g., Vercel) and point it to the Gradient backend. If cross-origin errors occur, add CORS to Flask (I can add this on request).

#### Train on Gradient Notebook (GPU)

1) Launch a Notebook with a GPU runtime.

2) Install dependencies: `pip install -r requirements.txt`

3) Capture data locally or upload `data/processed/`. Then run:

```
python train_model.py --data_dir data/processed --epochs 20 --batch_size 32 --lr 1e-3
```

4) Ensure `model/weights/best_model.pth` is present in your deployment image or persisted volume.

### CI/CD and GHCR image for this repo

Repo: https://github.com/Tharun16-R/Sign_Language

The included GitHub Actions workflow builds and publishes a container image to GHCR on pushes to `main`/`master`.

- Image name (auto):

```
ghcr.io/Tharun16-R/Sign_Language:latest
```

- If your default branch is `main`, after pushing, check the package at:
  https://github.com/Tharun16-R?tab=packages&repo_name=Sign_Language

- Make the package public if you want Gradient to pull it without credentials, or configure Gradient with GHCR credentials.

- In Gradient Deployment, select Image: `ghcr.io/Tharun16-R/Sign_Language:latest`, Port `8080`, Command `python app.py`.

### Windows local note

If you see "An attempt was made to access a socket in a way forbidden by its access permissions" on `8080`, another process may be bound or blocked:

- Change port: `set PORT=5000 && python app.py` and open `http://localhost:5000`.
- Or stop the conflicting app using port 8080.

## Notes

- Start with a small label set (A-Z or a few words). Collect 20-50 sequences per class with 30 frames each for a baseline.
- Tweak `SEQ_LEN`, model size, and training hyperparameters in `model/model.py` and `train_model.py`.
