#!/usr/bin/env python3
"""
store_roi_people_count_ranked.py

Same as previous resilient version, with one change:
 - For each frame, ROIs are ranked by their cumulative unique person counts.
 - Each ROI overlay now shows: Cur / Total / #M (rank among most) / #L (rank among least)
 - Prints short ranked summaries and final ranked lists at the end.
"""

import cv2, os, time, numpy as np
from ultralytics import YOLO

# -----------------------
# Simple config (edit here)
# -----------------------
SOURCE = r"storeTraffic.mp4"               # <-- set your input video file path here
OUTPUT = r"output_with_counts.mp4"
MODEL = "yolov8n.pt"                # or yolov8m.pt
CONF = 0.35
DEVICE = 'cpu'                        # "0" for GPU, "cpu" for CPU

# -----------------------
# ROI selection (same as before)
# -----------------------
rois = []
current_pts = []
drawing_window_name = "Select ROIs - Left click to add point (4), 'r' remove last ROI, 's' start"
finished_selection = False

def mouse_callback(event, x, y, flags, param):
    global current_pts, rois
    if finished_selection: return
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append((x, y))
        if len(current_pts) == 4:
            rois.append(current_pts.copy())
            current_pts.clear()

def draw_rois(img, rois, current_pts):
    overlay = img.copy()
    for idx, poly in enumerate(rois):
        pts = np.array(poly, np.int32).reshape((-1,1,2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0,200,255), thickness=2)
        cx = int(sum([p[0] for p in poly]) / len(poly))
        cy = int(sum([p[1] for p in poly]) / len(poly))
        cv2.putText(overlay, f"ROI_{idx+1}", (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    for p in current_pts:
        cv2.circle(overlay, p, 4, (0,255,0), -1)
    if len(current_pts) > 1:
        pts = np.array(current_pts, np.int32).reshape((-1,1,2))
        cv2.polylines(overlay, [pts], isClosed=False, color=(0,255,0), thickness=1)
    return overlay

# -----------------------
# Open video, select ROIs
# -----------------------
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {SOURCE}")

ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise SystemExit("Cannot read first frame (empty/corrupt video)")

# scale down for ROI selection if very wide
scale = 1.0
max_w = 1280
if first_frame.shape[1] > max_w:
    scale = max_w / first_frame.shape[1]
    first_frame_small = cv2.resize(first_frame, (0,0), fx=scale, fy=scale)
else:
    first_frame_small = first_frame.copy()

cv2.namedWindow(drawing_window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(drawing_window_name, mouse_callback)

print("== ROI Selection Instructions ==")
print(" - Left click to add points (4 clicks per ROI)")
print(" - 'u' undo last point, 'r' remove last ROI, 's' start inference, 'q' quit")

while True:
    vis = draw_rois(first_frame_small.copy(), rois, current_pts)
    cv2.imshow(drawing_window_name, vis)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('s'):
        finished_selection = True
        break
    elif k == ord('q'):
        print("Quitting (no processing).")
        cap.release(); cv2.destroyAllWindows()
        raise SystemExit()
    elif k == ord('r'):
        if rois:
            rois.pop(); print("Removed last ROI.")
    elif k == ord('u'):
        if current_pts:
            current_pts.pop(); print("Undid last point.")

cv2.destroyWindow(drawing_window_name)

# scale ROI coords back to original if we used scale
if scale != 1.0:
    inv_scale = 1.0 / scale
    rois = [[(int(x*inv_scale), int(y*inv_scale)) for (x,y) in poly] for poly in rois]

if len(rois) == 0:
    cap.release()
    raise SystemExit("No ROIs selected. Exiting.")

print(f"{len(rois)} ROIs selected. Starting inference...")

# -----------------------
# Model + tracker fallback logic
# -----------------------
model = YOLO(MODEL)

# tracker candidate list: try yaml first (most ultralytics expect this), then short name, then no tracker
tracker_candidates = ["bytetrack.yaml", "bytetrack", None]

track_base_kwargs = {"conf": CONF, "classes": 0, "device": DEVICE}

def point_in_poly(pt, poly):
    contour = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0

# We'll attempt each tracker option until one works
results_stream = None
used_tracker = None
last_exception = None
for t in tracker_candidates:
    try:
        kwargs = dict(track_base_kwargs)
        if t is not None:
            kwargs["tracker"] = t
            print(f"Trying tracker='{t}'")
        else:
            print("Trying without explicit tracker (default).")
        # Note: model.track returns a generator when stream=True
        results_stream = model.track(source=SOURCE, stream=True, **kwargs)
        used_tracker = t
        break
    except Exception as e:
        last_exception = e
        print(f"Tracker option '{t}' failed with: {e}")

if results_stream is None:
    cap.release()
    raise SystemExit("All tracker attempts failed. Last error:\n" + str(last_exception) + 
                     "\nTry updating ultralytics: pip install -U ultralytics")

# -----------------------
# Iterate frames, create writer only after first frame
# -----------------------
roi_seen_ids = {i: set() for i in range(len(rois))}
roi_current_ids = {i: set() for i in range(len(rois))}
out_writer = None
frame_idx = 0
start_t = time.time()
print("Processing frames (tracker used: {})...".format(used_tracker if used_tracker is not None else "default"))

try:
    for result in results_stream:
        frame_idx += 1
        frame = result.orig_img if hasattr(result, "orig_img") else result.orig_img

        # create writer lazily with correct frame size/fps
        if out_writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (w, h))
            if not out_writer.isOpened():
                raise SystemExit(f"Failed to open VideoWriter for {OUTPUT}")

        # clear current ids this frame
        for k in roi_current_ids: roi_current_ids[k].clear()

        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # xyxy extraction safe for tensor or list
            try:
                xyxy = boxes.xyxy.cpu().numpy()
            except:
                xyxy = np.array(boxes.xyxy)
            # ids: tracked id if available
            ids = None
            if hasattr(boxes, "id") and boxes.id is not None:
                try:
                    ids = boxes.id.cpu().numpy().astype(int)
                except:
                    ids = np.array(boxes.id).astype(int)
            else:
                ids = np.arange(len(xyxy)) + 1

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = box[:4]
                tid = int(ids[i])
                # Use bottom-center of bbox to check feet presence in ROI
                cx = int((x1 + x2) / 2.0)
                cy = int(y2)  # bottom y coordinate of bbox
                for ridx, poly in enumerate(rois):
                    if point_in_poly((cx, cy), poly):
                        roi_current_ids[ridx].add(tid)
                        roi_seen_ids[ridx].add(tid)
                # draw bbox + id
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                # draw bottom-center marker
                cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)

        # ----- Ranking logic (per-frame) -----
        # cumulative counts per ROI
        cumul_counts = {ridx: len(roi_seen_ids[ridx]) for ridx in range(len(rois))}
        # rank most -> least (1 = most)
        sorted_desc = sorted(cumul_counts.items(), key=lambda x: x[1], reverse=True)
        rank_most = {}
        for rank, (ridx, cnt) in enumerate(sorted_desc, start=1):
            rank_most[ridx] = rank
        # rank least -> most (1 = least)
        sorted_asc = sorted(cumul_counts.items(), key=lambda x: x[1])
        rank_least = {}
        for rank, (ridx, cnt) in enumerate(sorted_asc, start=1):
            rank_least[ridx] = rank

        # small printed summary (top3 most and top3 least)
        if frame_idx % 50 == 0:  # print every 50 frames to avoid spam
            top3 = sorted_desc[:3]
            bottom3 = sorted_asc[:3]
            print(f"[Frame {frame_idx}] Top3 most (ROI, count): {top3}")
            print(f"[Frame {frame_idx}] Top3 least (ROI, count): {bottom3}")

        # draw ROIs and overlays (including ranks)
        for ridx, poly in enumerate(rois):
            pts = np.array(poly, np.int32).reshape((-1,1,2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0,200,255), thickness=2)
            cumul = cumul_counts[ridx]; current = len(roi_current_ids[ridx])
            rm = rank_most.get(ridx, "-"); rl = rank_least.get(ridx, "-")
            label = f"ROI_{ridx+1} Cur:{current} Tot:{cumul} #M:{rm} #L:{rl}"
            lx, ly = poly[0]
            cv2.putText(frame, label, (lx+5, ly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            id_text = ",".join(str(i) for i in sorted(list(roi_current_ids[ridx]))[:10])
            if id_text: cv2.putText(frame, id_text, (lx+5, ly+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

        # write frame & display
        if out_writer is not None:
            out_writer.write(frame)
        cv2.imshow("Inference (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted.")
            break

    elapsed = time.time() - start_t
    if frame_idx > 0:
        print(f"Done. Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx/elapsed:.2f} FPS)")
    else:
        print("No frames processed by model.track.")

except Exception as e:
    print("Error during processing:", e)

finally:
    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

# If nothing was written, remove output file to avoid empty/corrupt video
if (not os.path.exists(OUTPUT)) or (os.path.exists(OUTPUT) and os.path.getsize(OUTPUT) == 0) or frame_idx == 0:
    if os.path.exists(OUTPUT):
        try:
            os.remove(OUTPUT)
            print(f"Removed empty/corrupt output file: {OUTPUT}")
        except Exception:
            pass
    print("No valid output produced. Check tracker/ultralytics version (try: pip install -U ultralytics) and that model weights are available.")
    raise SystemExit()

# Final ranked summary
final_counts = {ridx: len(roi_seen_ids[ridx]) for ridx in range(len(rois))}
final_sorted_desc = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
final_sorted_asc = sorted(final_counts.items(), key=lambda x: x[1])
print("Final per-ROI unique person counts:")
for ridx in range(len(rois)):
    print(f"ROI_{ridx+1}: unique_persons = {final_counts[ridx]}")

if len(rois) >= 3:
    print("\nTop 3 ROIs by unique person count:")
    for ridx, cnt in final_sorted_desc[:3]:
        print(f"  ROI_{ridx+1}: {cnt}")
    print("\nBottom 3 ROIs by unique person count:")
    for ridx, cnt in final_sorted_asc[:3]:
        print(f"  ROI_{ridx+1}: {cnt}")

       
# print("\nROIs ordered most -> least (ROI, count):")
# print(final_sorted_desc)
# print("\nROIs ordered least -> most (ROI, count):")
# print(final_sorted_asc)
# print(f"\nOutput saved to {os.path.abspath(OUTPUT)}")
