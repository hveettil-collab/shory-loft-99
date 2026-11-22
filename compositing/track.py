"""Pass 2: propagate display quads from confident anchors with SIFT+RANSAC homographies.
Segments are the physical shots of the reference; matching never crosses a segment boundary."""
import sys, json, cv2, numpy as np
frames_dir, det_json, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
SEGMENTS = json.load(open(sys.argv[4]))  # [[start,end,label], ...]
det = json.load(open(det_json))
sift = cv2.SIFT_create(4000)
cache = {}
def load(i):
    if i not in cache:
        cache[i] = cv2.cvtColor(cv2.imread(f"{frames_dir}/f{i:03d}.png"), cv2.COLOR_BGR2GRAY)
    return cache[i]
def conf(i):
    c = det.get(str(i), [])
    return c and c[0]["fill"] > 0.9
def feats(i, quad, grow=1.12):
    g = load(i); m = np.zeros_like(g)
    q = np.array(quad, np.float32); c = q.mean(0)
    cv2.fillConvexPoly(m, ((q - c) * grow + c).astype(np.int32), 255)
    return sift.detectAndCompute(g, m)
def homog(a, qa, b, qb_guess):
    ka, da = feats(a, qa); kb, db = feats(b, qb_guess, 1.35)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return None, 0
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 12: return None, len(good)
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.5)
    return H, int(inl.sum()) if inl is not None else 0
out = {}
for s, e, label in SEGMENTS:
    anchors = [i for i in range(s, e + 1) if conf(i)]
    if not anchors:
        print("segment", s, e, label, "no anchors"); continue
    for i in range(s, e + 1):
        if conf(i):
            out[i] = {"quad": det[str(i)][0]["quad"], "src": "detect", "inliers": -1}
    # propagate outward from anchors, chaining guesses but always matching to nearest anchor
    for direction in (1, -1):
        rng = range(s, e + 1) if direction == 1 else range(e, s - 1, -1)
        last_anchor = None; last_quad = None
        for i in rng:
            if conf(i):
                last_anchor, last_quad = i, det[str(i)][0]["quad"]; continue
            if last_anchor is None: continue
            H, n = homog(last_anchor, det[str(last_anchor)][0]["quad"], i, last_quad)
            if H is None or n < 15: continue
            q = cv2.perspectiveTransform(np.float32([det[str(last_anchor)][0]["quad"]]), H)[0].tolist()
            prev = out.get(i)
            dist = abs(i - last_anchor)
            if prev is None or (prev["src"] == "track" and dist < prev["dist"]):
                out[i] = {"quad": q, "src": "track", "inliers": n, "dist": dist, "anchor": last_anchor}
            last_quad = q
missing = [i for i in range(1, 361) if i not in out]
print("tracked frames:", len(out), "missing:", missing)
json.dump(out, open(out_json, "w"))
