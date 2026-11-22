# Shory × Duo — screen-replacement compositing pipeline

Source footage: `original_iphone_duo_reference.mp4` (720×1280, 30 fps, 360 frames) from the Shory exact-compositing kit.
Hardware, hands and camera are never regenerated — only pixels inside the display glass are replaced.

## Status
| Step | State |
|---|---|
| Frame extraction (360/360) | done |
| Display-quad detection, clean frames (black-bezel hole) | done — 156 frames |
| Laptop-mode upper panel (content-colour detector) | done — 56/60 frames |
| Anchor-based SIFT+RANSAC propagation (skin-excluded) | 248 frames |
| Bezel-ring KLT tracker for content-changing stretches | f1–31 verified (hand pass + iOS blur); f33–36 is the unfold, needs per-panel tracking |
| Fold transitions (f37–50, f139–156, f187–192, f286–300) | not started — need per-panel tracking |
| Finger/hand mattes | not started |
| Removal of source captions ("Hello.", "Hello, hello.", "Apple TV…", Apple-logo "iPhone Duo") | not started |
| **Figma UI exports** | **BLOCKED** — `www.figma.com` (Figma MCP asset host) is denied by this environment's network policy |

## Figma frames used (file `dHgah0alsmzaJq1L6BiteG`)
- `94:803` Duo / Home, closed · Dark (466×678) → outer display, segments `closed_outer*`
- `82:683` Duo / Home, open · Dark (951×669) → inner display held landscape
- `140:768` Duo / Home, open portrait · Dark (669×951) → inner display held portrait, and its halves on the tent / laptop panels

## Scripts
- `detect_screens.py frames_dir out.json`
- `detect_laptop.py frames_dir out.json start end`
- `track.py frames_dir detect.json out.json segments.json`
- `viz.py frames_dir quads.json out.png step start end`
- `segments.json` — shot map measured from the footage (the kit README's shot map does not match the footage).
