# Development guide

How the project works, how to change it, and how to put a change live.

- [How it works](#how-it-works)
- [Run it locally](#run-it-locally)
- [Change something](#change-something)
- [Ship a change](#ship-a-change)
- [Upgrade the model](#upgrade-the-model)
- [Traps and fixes](#traps-and-fixes)

---

## How it works

Four moving parts. A browser (or any HTTP client) sends an image, a video, or a
stream of webcam frames to a **FastAPI** server. The server hands the pixels to a
**YOLO11** model for detection. For anything with motion it also runs
**ByteTrack**, which gives each vehicle an identity that survives across frames.
Results come back as JSON; for video, an annotated MP4 is encoded with **ffmpeg**.

### The one idea that matters

Detection answers *"what is in this frame."* That is not enough to count traffic —
a car sitting in view for 90 frames would be counted 90 times. Tracking answers
*"is that the same car I saw a second ago,"* and only with a stable identity can
you count each vehicle **once** as it crosses a line.

That line is a virtual tripwire, and it is the primitive real traffic
installations are built on. Everything else is plumbing around those two ideas.

### Where the code lives

| Path | Responsibility |
|---|---|
| `app/config.py` | Every setting, read from environment variables |
| `app/detector.py` | Single-image detection — no memory between calls |
| `app/video.py` | Tracking, line counting, ffmpeg encoding |
| `app/jobs.py` | Background video jobs and TTL cleanup |
| `app/main.py` | API routes (REST + WebSocket) |
| `app/static/index.html` | The browser console — one file, no build step |
| `scripts/stream.py` | Headless runner for a real RTSP camera |
| `scripts/train.py` | Fine-tuning entry point |
| `tests/` | 64 tests |

### The three modes

| Mode | Endpoint | Runs |
|---|---|---|
| Image | `POST /api/detect` | Detection only. Stateless. |
| Video | `POST /api/video` | Detection + tracking. Async job — a 30 s clip takes ~40 s. |
| Live | `WS /ws/live` | Detection + tracking, one frame at a time. |

Live uses a WebSocket because each frame needs a reply before the next is sent.
That single rule is the backpressure: without it a 60 Hz camera queues frames
faster than the CPU can clear them and the overlay drifts seconds behind reality.

---

## Run it locally

### Python — the fast loop

```bash
uvicorn app.main:app --reload --port 7860
```

Open <http://localhost:7860>. Editing anything under `app/` restarts the server
automatically. Spend most of your time here.

```bash
python -m pytest
```

45 tests, about 15 seconds. Run before every push.

### Docker — what actually deploys

```bash
docker compose up --build
```

Builds the same image Hugging Face builds. Slower, but the only way to catch
problems that exist only on Linux. Stop with `docker compose down`.

Docker Desktop must be running first — its Linux VM takes a few minutes to start.

### The directory problem

A fresh PowerShell window opens in `C:\Windows\system32`, which is why both
`not a git repository` and `no configuration file provided` show up. Git has a
flag that makes location irrelevant:

```bash
git -C "D:\Traffic Object Detection model" status
```

Docker has no equivalent — `cd` to the project first.

---

## Change something

### Detection thresholds

The console sliders do this live. To change the default, set `DEFAULT_CONF` /
`DEFAULT_IOU` as environment variables, or edit `app/config.py`.

Lower confidence catches distant vehicles but invents false ones. There is no
correct value; it is a trade you tune per camera.

### Which classes are detected

Edit `TRAFFIC_CLASS_NAMES` in `app/config.py`. Names must match COCO's exactly.
Add a colour for any new class in `CLASS_COLORS` just below.

The console builds its filter chips from `/api/classes`, so adding a name here
makes it appear in the UI with no front-end change.

### The counting line

The height slider moves it. To make it diagonal, pass different start/end points
to `LineCounter` in `app/video.py` — it already accepts two arbitrary points in
normalised (0–1) coordinates; only the API hard-codes a horizontal line.

Counting works by testing which side of the line a vehicle's centre is on and
noticing when that flips. That maths works for any line, so a diagonal is a
change to the *caller*, not the counter.

### Let the user draw the line (recommended next feature)

Capture two clicks on the canvas in `app/static/index.html`, convert to 0–1
coordinates, and send them instead of `line_y`. The back end needs a new form
field for the two points, passed through to `LineCounter`.

Real cameras are never aimed so the lanes sit neatly under a horizontal line.
This is the single change that makes the demo look like a product — and it
touches the front end, the API, and `video.py`, so it is a good way to learn the
whole stack.

### Accessibility, and why the palette is what it is

The colour tokens are not aesthetic guesses; they were measured against WCAG 2.1
AA and three pairs failed:

| Pair | Was | Needed | Now |
|---|--:|--:|--:|
| `--dim` on `--panel` (hint text) | 2.52:1 | 4.5 | 5.01 |
| `--dim` on `--bg` (footer) | 2.72:1 | 4.5 | 5.41 |
| `--line-2` on `--panel` (control borders) | 1.47:1 | 3.0 | 3.08 |

The naive fix — lighten `--dim` until it passes — would have pushed it onto
`--muted` and collapsed the three-tier text hierarchy into two. The whole scale
moved up instead: `--muted` took a lighter value and `--dim` inherited the old
`--muted`, so the tiers stay distinct *and* all three clear 4.5:1.

`--line-2` needs only 3:1 rather than 4.5:1 because it is a non-text UI boundary
(WCAG 1.4.11). It qualifies because it draws slider tracks, ghost buttons and the
drop zone — things whose edge *is* the affordance — not because it is a border.

If you change a colour, re-check it. The ratio is
`(L_lighter + 0.05) / (L_darker + 0.05)` on relative luminance.

### Touch targets

The layout is deliberately dense, which is right with a mouse and wrong with a
thumb. Measured on a 390px viewport, 23 controls were under the 44px minimum —
class chips at 29px, "Clear all" at 13px, and sliders whose entire hit area was
the 4px visual track.

Rather than inflate everything and lose the density, `@media (pointer: coarse)`
expands them only where the pointer actually is coarse. Sliders keep their thin
look by centring the 4px gradient inside a 44px grab area via `background-size`.

If you add a control, check it at 390px before shipping.

### The look of the page

Everything is in `app/static/index.html` — styles at the top, logic at the
bottom. Colours are CSS variables in the `:root` block; change those and the
whole console follows. No npm, no bundler, nothing to break.

### A new API endpoint

Add a function in `app/main.py` with a decorator like
`@app.get("/api/yourthing")`. It appears in the interactive docs at `/docs`
automatically, because FastAPI generates them from your type hints. Add a test.

### Point it at a real camera

```bash
python scripts/stream.py --source "rtsp://user:pass@10.0.0.20:554/stream1" --csv counts.csv
```

Works with an RTSP URL, a video file, or `--source 0` for a webcam. Add
`--headless` on a machine with no display. This is the deployment story — the web
console is the demo, this script is what runs beside a camera.

### The rule that keeps you out of trouble

**Change one thing, run the tests, commit.** The tests exist so you can be brave.
Green means you did not break the API contract, the counting maths, or the video
encoder. Red means read the failing test name before touching anything.

---

## Ship a change

Two remotes: `origin` is GitHub (the code people read), `hf` is Hugging Face
(the demo people click). Both get the same commits.

```bash
python -m pytest
git add -A
git commit -m "Short description of what changed"
git push origin main
git push hf main
```

Pushing to `hf` rebuilds the Space — watch the **Logs** tab. A build takes
roughly 5–15 minutes.

**Commit messages:** first line says what changed, under ~70 characters, in the
imperative ("Add speed estimation", not "Added"). If the change is not obvious,
leave a blank line and explain *why* underneath.

**`--force`** is only needed after rewriting history that was already pushed. A
normal push rejected with *"remote contains work that you do not have locally"*
means either you rewrote history, or someone else pushed — and forcing would
delete their work.

**Never `git pull` when a push is rejected**, even though git's hint suggests it.
Here it would merge Hugging Face's placeholder README into your history.

---

## Upgrade the model

### The cheap upgrade

`yolo11s` is the default. `yolo11n` is smaller and faster; `yolo11m` catches
more distant vehicles at the cost of speed. No code change:

```bash
MODEL_PATH=yolo11s.pt uvicorn app.main:app --port 7860
```

On the free Hugging Face tier, medium will likely be too slow for live mode.

### The real upgrade: fine-tune

The served model is trained on COCO, a general photo dataset. It has never seen a
traffic camera's angle, lighting, or weather. Fine-tuning is what lets you
publish an accuracy number.

**Not on this machine** — there is no CUDA GPU here, so training would take days.
Use Google Colab's free GPU.

1. Get a labelled traffic dataset — BDD100K, KITTI, or Roboflow Universe. Do not
   label from scratch. Check the licence before redistributing anything derived.
2. Arrange it per [DATASET.md](DATASET.md). The `images/` and `labels/` trees must
   mirror each other exactly.
3. Update the class list in `dataset/data.yaml`.
4. In Colab with a GPU runtime:
   ```bash
   python scripts/train.py --model yolo11s.pt --epochs 100 --batch 16 --device 0
   ```
5. Download `best.pt` into `weights/`.
6. Serve it:
   ```bash
   MODEL_PATH=weights/traffic-yolo.pt uvicorn app.main:app --port 7860
   ```
7. Put the reported **mAP50-95** in the README and delete the "no accuracy
   metrics are published" limitation.

Swapping models needs no code change because every setting is an environment
variable, and the class filter falls back to "keep everything" when a custom
model's labels do not match COCO's names.

### Weights are not in git

`.gitignore` excludes `*.pt` — model files are tens to hundreds of megabytes and
GitHub rejects anything over 100 MB. The container downloads `yolo11s.pt` during
its build. If you fine-tune, publish the weights as a Hugging Face model repo and
have the app download them rather than committing them.

---

## Traps and fixes

Every failure hit while building this, and what actually caused it.

| What you see | Real cause | Fix |
|---|---|---|
| `not a git repository` | Fresh terminal opens in `system32` | `cd` to the project, or use `git -C "path"` |
| `no configuration file provided` | Docker looks for `docker-compose.yml` in the current folder | `cd` to the project first |
| Password shows nothing while typing | Not a bug — terminals hide password input by design | Type or paste blindly, press Enter |
| `libxcb.so.1: cannot open shared object file` | Ultralytics pulls in the GUI build of OpenCV, which needs X libraries a slim container has no reason to carry | Fixed in the Dockerfile — it removes the GUI build and reinstalls headless |
| `cv2 has no attribute __version__` | Both OpenCV packages unpack into the same folder; removing one guts the other | Reinstall headless after uninstalling — the Dockerfile does this |
| HF: `push rejected … contains binary files` | Hugging Face requires binaries in Git LFS | Already migrated; `.gitattributes` picks up new images and clips automatically |
| A JPEG is 130 bytes of text | It is an LFS pointer, not the file | `git lfs pull` |
| Docker Desktop "unable to start" | Its Linux VM has not been provisioned | Quit fully and reopen. Never "Reset to factory defaults" — that wipes all your images |
| Live camera does nothing in a background tab | Browsers pause animation frames on hidden tabs | Intended. The pill reads "paused · tab hidden" and resumes when you return |
| Space builds, then crashes on boot | Something works on Windows but not Linux | Reproduce with `docker compose up --build` before pushing |

### Reading a failed build

On the Space's **Logs** tab, scroll to the *first* red line, not the last. Errors
cascade, so the bottom of the log is usually a consequence rather than the cause.
The line that matters names a file and a reason.

---

## Design decisions worth knowing

Three things that are not obvious from reading the code, and that a reviewer is
likely to ask about.

**One model per tracking session.** Ultralytics stores tracker state on the model
object, and `persist=True` keeps it between calls. That makes the model
single-tenant: two sessions sharing one inherit each other's frame counter and
open tracks, so the second silently loses its first frame and its counts are
polluted by the first session's vehicles. Each session gets its own instance from
`detector.new_tracking_model()`. Concurrent live sessions are capped so this
cannot grow without bound.

**ffmpeg, not `cv2.VideoWriter`.** The codecs OpenCV can reach in a slim
container (`mp4v`) produce files Chrome and Safari refuse to play inline.
`imageio-ffmpeg` ships a static ffmpeg with libx264 on every platform.

**The inference pool belongs to the app lifespan, not the module.** A
module-level `ThreadPoolExecutor` is shut down by the first app teardown and can
never be used again, which breaks any second startup in the same process.
