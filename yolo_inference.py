import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter

# =========================================================
# CONFIG
# =========================================================

INPUT_VIDEO = "input_videos/messi.mp4"
OUTPUT_VIDEO = "output_videos/detected.mp4"

# =========================================================
# TEAM COLORS HSV
# =========================================================

TEAM_COLORS = {

    # Argentina
    "Argentina": {

        # Light blue stripes
        "blue": [
            (75, 30, 40),
            (140, 255, 255)
        ],

        # White jersey
        "white": [
            (0, 0, 140),
            (180, 60, 255)
        ],

        # Black shorts
        "black": [
            (0, 0, 0),
            (180, 255, 80)
        ],
    },

    # Austria
    "Austria": {

        "red1": [
            (0, 120, 70),
            (10, 255, 255)
        ],

        "red2": [
            (170, 120, 70),
            (180, 255, 255)
        ],

        "white": [
            (0, 0, 170),
            (180, 40, 255)
        ],
    },
}

# =========================================================
# DRAW COLORS
# =========================================================

TEAM_COLORS_BGR = {
    "Argentina": (255, 200, 0),
    "Austria": (0, 0, 255),
    "Unknown": (120, 120, 120),
}

# =========================================================
# LOAD MODEL
# =========================================================

# Better:
# yolov8m.pt
# yolov8x.pt
# yolov8x-seg.pt

model = YOLO("yolov8x.pt")

# =========================================================
# TRACK HISTORY
# =========================================================

track_history = defaultdict(list)

# =========================================================
# HELPERS
# =========================================================

def count_color_pixels(hsv_img, lower, upper):

    mask = cv2.inRange(
        hsv_img,
        np.array(lower),
        np.array(upper)
    )

    return cv2.countNonZero(mask)

# =========================================================
# CROP PLAYER REGIONS
# =========================================================

def crop_regions(frame, box):

    x1, y1, x2, y2 = map(int, box)

    h = y2 - y1
    w = x2 - x1

    # center crop
    margin = int(w * 0.15)

    x1 += margin
    x2 -= margin

    # safety
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)

    y1 = max(0, y1)
    y2 = min(frame.shape[0], y2)

    # =====================================================
    # JERSEY
    # =====================================================

    jersey_top = y1 + int(h * 0.20)
    jersey_bottom = y1 + int(h * 0.55)

    # =====================================================
    # SHORTS
    # =====================================================

    shorts_top = y1 + int(h * 0.55)
    shorts_bottom = y1 + int(h * 0.85)

    jersey = frame[
        jersey_top:jersey_bottom,
        x1:x2
    ]

    shorts = frame[
        shorts_top:shorts_bottom,
        x1:x2
    ]

    return jersey, shorts

# =========================================================
# CLASSIFY TEAM
# =========================================================

def classify_team(frame, box):

    jersey, shorts = crop_regions(frame, box)

    if jersey.size == 0 or shorts.size == 0:
        return "Unknown", 0.0

    # =====================================================
    # DENOISE
    # =====================================================

    jersey = cv2.GaussianBlur(
        jersey,
        (5, 5),
        0
    )

    shorts = cv2.GaussianBlur(
        shorts,
        (5, 5),
        0
    )

    # =====================================================
    # HSV
    # =====================================================

    hsv_jersey = cv2.cvtColor(
        jersey,
        cv2.COLOR_BGR2HSV
    )

    hsv_shorts = cv2.cvtColor(
        shorts,
        cv2.COLOR_BGR2HSV
    )

    total_j = jersey.shape[0] * jersey.shape[1]
    total_s = shorts.shape[0] * shorts.shape[1]

    # =====================================================
    # ARGENTINA
    # White dominant + some blue
    # =====================================================

    blue_pixels = count_color_pixels(
        hsv_jersey,
        *TEAM_COLORS["Argentina"]["blue"]
    )

    white_pixels_arg = count_color_pixels(
        hsv_jersey,
        *TEAM_COLORS["Argentina"]["white"]
    )

    black_pixels = count_color_pixels(
        hsv_shorts,
        *TEAM_COLORS["Argentina"]["black"]
    )

    blue_ratio = blue_pixels / total_j
    white_ratio = white_pixels_arg / total_j
    black_ratio = black_pixels / total_s

    argentina_score = 0

    # White jersey dominant
    if white_ratio > 0.18:
        argentina_score += white_ratio * 2.5

    # Light blue exists
    if blue_ratio > 0.05:
        argentina_score += blue_ratio * 4.0

    # Black shorts bonus
    if black_ratio > 0.10:
        argentina_score += black_ratio * 1.0

    # =====================================================
    # AUSTRIA
    # =====================================================

    red_pixels = (

        count_color_pixels(
            hsv_jersey,
            *TEAM_COLORS["Austria"]["red1"]
        )

        +

        count_color_pixels(
            hsv_jersey,
            *TEAM_COLORS["Austria"]["red2"]
        )
    )

    white_pixels_aut = count_color_pixels(
        hsv_shorts,
        *TEAM_COLORS["Austria"]["white"]
    )

    red_ratio = red_pixels / total_j
    white_ratio_aut = white_pixels_aut / total_s

    austria_score = 0

    if red_ratio > 0.10:
        austria_score += red_ratio * 5.0

    if white_ratio_aut > 0.10:
        austria_score += white_ratio_aut * 1.5

    # =====================================================
    # DEBUG
    # =====================================================

    print(
        f"[Argentina] "
        f"Blue:{blue_ratio:.2f} "
        f"White:{white_ratio:.2f} "
        f"Black:{black_ratio:.2f} "
        f"Score:{argentina_score:.2f}"
    )

    print(
        f"[Austria] "
        f"Red:{red_ratio:.2f} "
        f"White:{white_ratio_aut:.2f} "
        f"Score:{austria_score:.2f}"
    )

    # =====================================================
    # FILTER LOW SIGNAL
    # =====================================================

    if (
        red_ratio < 0.03 and
        blue_ratio < 0.03 and
        white_ratio < 0.10
    ):
        return "Unknown", 0.0

    # =====================================================
    # FINAL SCORES
    # =====================================================

    scores = {
        "Argentina": argentina_score,
        "Austria": austria_score
    }

    best_team = max(
        scores,
        key=scores.get
    )

    confidence = scores[best_team]

    # Lower threshold
    if confidence < 0.10:
        return "Unknown", confidence

    return best_team, round(confidence, 2)

# =========================================================
# TEMPORAL VOTING
# =========================================================

def stabilize_team_prediction(track_id, current_team):

    if track_id is None:
        return current_team

    track_history[track_id].append(current_team)

    # keep last 20 predictions
    if len(track_history[track_id]) > 20:
        track_history[track_id].pop(0)

    votes = track_history[track_id]

    stable_team = Counter(votes).most_common(1)[0][0]

    return stable_team

# =========================================================
# DRAW
# =========================================================

def draw_player(frame, box, team, confidence, track_id=None):

    x1, y1, x2, y2 = map(int, box)

    color = TEAM_COLORS_BGR.get(
        team,
        (120, 120, 120)
    )

    # box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # label
    id_text = (
        f"#{track_id} "
        if track_id is not None
        else ""
    )

    label = f"{id_text}{team} ({confidence:.2f})"

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )

    cv2.rectangle(
        frame,
        (x1, y1 - th - 6),
        (x1 + tw + 6, y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

# =========================================================
# MAIN
# =========================================================

def process_video():

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_count = 0

    team_count = defaultdict(int)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # =================================================
        # YOLO TRACKING
        # =================================================

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.45,
            iou=0.5,
            verbose=False
        )

        if results[0].boxes is not None:

            boxes = results[0].boxes

            for box in boxes:

                xyxy = box.xyxy[0].cpu().numpy()

                det_conf = float(
                    box.conf[0]
                )

                track_id = (
                    int(box.id[0])
                    if box.id is not None
                    else None
                )

                # classify
                team, team_conf = classify_team(
                    frame,
                    xyxy
                )

                # stabilize
                team = stabilize_team_prediction(
                    track_id,
                    team
                )

                # draw
                draw_player(
                    frame,
                    xyxy,
                    team,
                    team_conf,
                    track_id
                )

                team_count[team] += 1

        # =================================================
        # SCOREBOARD
        # =================================================

        y_pos = 30

        for team, count in team_count.items():

            color = TEAM_COLORS_BGR.get(
                team,
                (255, 255, 255)
            )

            cv2.putText(
                frame,
                f"{team}: {count}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            y_pos += 30

        # =================================================
        # WRITE
        # =================================================

        out.write(frame)

        if frame_count % 30 == 0:

            print(
                f"✅ Processed frame: "
                f"{frame_count}"
            )

    cap.release()
    out.release()

    print("\n🎬 Finished.")
    print(f"📁 Saved to: {OUTPUT_VIDEO}")
    print(f"📊 Final Count: {dict(team_count)}")

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    process_video()