#!/usr/bin/env python3
"""
detector.py

NXP CUP India 2026 - Autonomous Medical Response Challenge
Standalone OpenCV prototype for reading the six-panel sign board.

The simulator always presents ONE large green board containing SIX fixed
panels laid out left to right as:

    +----+----+----+----+----+----+
    | A  | B  | C  | X  | Y  | Z  |
    | <- | <- | ^  | ^  | -> | -> |
    +----+----+----+----+----+----+

Column order never changes:
    column 0 -> A
    column 1 -> B
    column 2 -> C
    column 3 -> X
    column 4 -> Y
    column 5 -> Z

Because the letter-to-column mapping is fixed and known in advance, this
program never performs OCR or letter classification. Given a --goal letter,
it looks up the corresponding column directly, then analyzes ONLY the arrow
found in that column using classical geometric contour analysis. No AI /
deep-learning / OCR / template matching is used anywhere in this pipeline.

Usage:
    python detector.py --goal A --image SignBoard.png
    python detector.py --goal Z --image SignBoard.png --no-gui
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed, never-changing letter -> column mapping.
GOAL_COLUMN_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "X": 3,
    "Y": 4,
    "Z": 5,
}

NUM_COLUMNS = 6

# Size of the rectified (warped) board. Kept at a 3:1 width:height ratio so
# each of the six columns is a clean, evenly sized square-ish cell.
WARPED_WIDTH = 1200
WARPED_HEIGHT = 400

# HSV range used to segment the green board from the background.
# Fairly wide range to tolerate lighting variation in the simulator.
GREEN_HSV_LOWER = np.array([35, 40, 40])
GREEN_HSV_UPPER = np.array([90, 255, 255])

# Fraction of a column's height that belongs to the printed letter (ignored)
# vs. the arrow (analyzed).
LETTER_REGION_FRACTION = 0.40

# Navigation decision text for each detected arrow direction.
NAVIGATION_TABLE = {
    "LEFT": "TURN LEFT",
    "RIGHT": "TURN RIGHT",
    "STRAIGHT": "GO STRAIGHT",
}


# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------

def order_points(pts):
    """Order 4 (x, y) points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference

    return rect


def detect_board(image):
    """
    Locate the large green board in `image` using HSV thresholding,
    morphological cleanup, and largest-contour + polygon approximation.

    Returns:
        ordered_pts (4x2 float32 ndarray): corners of the board, ordered
            top-left, top-right, bottom-right, bottom-left.
        green_mask (ndarray): binary mask used for detection (debug).
        board_debug (ndarray): copy of `image` with the detected board
            polygon drawn on it (debug).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No green board detected in the image.")

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        raise RuntimeError("Largest green region is too small to be the board.")

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2)
    else:
        # Fallback: use the minimum-area rotated bounding rectangle so we
        # always end up with exactly 4 corner points.
        rect = cv2.minAreaRect(largest)
        quad = cv2.boxPoints(rect)

    ordered_pts = order_points(np.array(quad, dtype="float32"))

    board_debug = image.copy()
    cv2.polylines(
        board_debug,
        [ordered_pts.astype(np.int32)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3,
    )
    for (px, py) in ordered_pts:
        cv2.circle(board_debug, (int(px), int(py)), 6, (255, 0, 0), -1)

    return ordered_pts, mask, board_debug


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------

def warp_board(image, ordered_pts, width=WARPED_WIDTH, height=WARPED_HEIGHT):
    """Perspective-warp the quadrilateral defined by `ordered_pts` into a
    flat width x height rectangle."""
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped


# ---------------------------------------------------------------------------
# Column splitting
# ---------------------------------------------------------------------------

def split_into_columns(warped, num_columns=NUM_COLUMNS):
    """Split the warped board into `num_columns` equal-width vertical slices.

    Returns a list of (column_image, x_offset) tuples, x_offset being the
    column's left edge x-coordinate within the warped board (for debug
    drawing / coordinate translation).
    """
    height, width = warped.shape[:2]
    col_width = width // num_columns

    columns = []
    for i in range(num_columns):
        x_start = i * col_width
        # Last column absorbs any rounding remainder.
        x_end = width if i == num_columns - 1 else (i + 1) * col_width
        column_img = warped[0:height, x_start:x_end]
        columns.append((column_img, x_start))

    return columns


# ---------------------------------------------------------------------------
# Arrow ROI extraction
# ---------------------------------------------------------------------------

def extract_arrow_roi(column_img, letter_fraction=LETTER_REGION_FRACTION):
    """Crop the arrow region away from the panel borders."""
    height, width = column_img.shape[:2]

    top = int(height * 0.40)
    bottom = int(height * 0.95)
    left = int(width * 0.20)
    right = int(width * 0.80)

    arrow_roi = column_img[top:bottom, left:right]
    return arrow_roi, left, top


# ---------------------------------------------------------------------------
# Arrow direction detection
# ---------------------------------------------------------------------------

def _binarize_arrow(arrow_roi):
    """Convert the arrow ROI into a clean binary mask."""
    gray = cv2.cvtColor(arrow_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # The arrow is bright white against a darker green board, so plain
    # THRESH_BINARY (not INV) maps the arrow to white / background to
    # black. INV here was inverting the mask -- the background became
    # "foreground", which is why prior contour/mass analysis was
    # effectively analyzing everything EXCEPT the arrow.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return thresh


def _largest_arrow_contour(binary):
    """
    Return the arrow contour in the binary mask, filtering out thin
    full-height border/divider slivers first.

    A panel divider or border remnant that leaks into the ROI is almost
    always a thin, tall strip that spans nearly the entire ROI height. The
    real arrow blob never does this -- it's compact. Picking the largest
    contour by raw area alone can still grab the divider (a long thin
    contour can have a larger area than a small compact arrowhead), so
    reject any contour whose bounding-box height covers most of the ROI
    before ranking by area.
    """
    roi_h = binary.shape[0]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No arrow contour found in the selected column.")

    def is_border_sliver(c):
        _, _, cw, ch = cv2.boundingRect(c)
        return ch > 0.85 * roi_h and cw < 0.25 * binary.shape[1]

    candidates = [c for c in contours if not is_border_sliver(c)]
    if candidates:
        contours = candidates

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        raise RuntimeError("Arrow contour too small.")
    return contour


def _classify_arrow_by_mass(binary, contour, edge_fraction=0.4, center_gap=0.2):
    """
    Classify arrow direction using pixel-mass distribution instead of
    contour-tip geometry.

    Why: contour-tip / extension-ratio approaches broke whenever the wrong
    contour got selected (divider line, border sliver, etc.) or when the
    shaft's bounding box shape confused an aspect-ratio test. Mass
    distribution doesn't need a "clean" contour at all -- it just looks at
    how the white pixels in the WHOLE binary ROI are spread out, which is
    far more tolerant of blur, anti-aliasing, and minor segmentation noise.

    Logic:
      1. Decide horizontal (<- / ->) vs vertical (^) by comparing how many
         columns vs how many rows contain any white pixel at all.
      2. For horizontal arrows, compare the pixel mass in the outer-left
         `edge_fraction` of the width against the outer-right
         `edge_fraction`, ignoring a `center_gap` band in the middle (the
         shaft contributes roughly equal, symmetric mass to both sides, so
         excluding it sharpens the head-vs-no-head signal). Whichever side
         has more mass has the arrowhead -> that's the direction.
      3. Vertical spread wins -> STRAIGHT (this board only has ^ vertically).
    """
    if binary.size == 0 or cv2.countNonZero(binary) == 0:
        raise RuntimeError("Empty arrow mask; nothing to classify.")

    # Restrict the mass analysis to the arrow's own bounding box (from the
    # largest contour) rather than the full ROI. The full ROI can still
    # contain a stray pixel or two from panel borders near the top/bottom
    # edges of the crop; those get counted as "spread" across the whole
    # height even though they have nothing to do with the arrow. Cropping
    # to the bbox first keeps the mass-distribution idea (no tip-finding)
    # while eliminating that noise source.
    bx, by, bw, bh = cv2.boundingRect(contour)
    pad = 2
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(binary.shape[1], bx + bw + pad)
    y1 = min(binary.shape[0], by + bh + pad)
    binary = binary[y0:y1, x0:x1]

    h, w = binary.shape[:2]
    if h == 0 or w == 0 or cv2.countNonZero(binary) == 0:
        raise RuntimeError("Empty arrow mask after cropping to bbox.")

    col_sum = (binary > 0).sum(axis=0).astype(np.float64)  # per-column pixel count
    row_sum = (binary > 0).sum(axis=1).astype(np.float64)  # per-row pixel count

    horizontal_spread = np.count_nonzero(col_sum)
    vertical_spread = np.count_nonzero(row_sum)

    # Outer bands used for left/right mass comparison, skipping the middle
    # `center_gap` fraction of the width where the shaft lives.
    edge_px = max(1, int(round(w * edge_fraction)))
    gap_px = int(round(w * center_gap / 2))
    mid = w // 2

    left_band = col_sum[:edge_px]
    right_band = col_sum[max(0, w - edge_px):]
    # Drop overlap with the central gap band so a small ROI doesn't
    # double-count shaft pixels near the middle.
    if edge_px > mid - gap_px:
        left_band = col_sum[: max(1, mid - gap_px)]
        right_band = col_sum[min(w - 1, mid + gap_px):]

    left_mass = float(left_band.sum())
    right_mass = float(right_band.sum())

    # Orientation: use the arrow's bounding-box aspect ratio, not raw
    # spread counts. An up-arrow is genuinely tall and narrow (bh >> bw).
    # A left/right arrow's triangular head still adds real vertical
    # extent, so height alone can rival the width -- a plain spread-count
    # comparison misclassifies it as STRAIGHT. Aspect ratio is decisive
    # here because the font/size is fixed across the whole board.
    is_tall_narrow = h > 1.4 * w
    if is_tall_narrow:
        direction = "STRAIGHT"
    else:
        direction = "LEFT" if left_mass > right_mass else "RIGHT"

    # Debug/visualization info (kept compatible with _draw_arrow_debug).
    x, y, bw, bh = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
    else:
        cx, cy = x + bw // 2, y + bh // 2

    points = contour.reshape(-1, 2)
    leftmost = tuple(points[np.argmin(points[:, 0])])
    rightmost = tuple(points[np.argmax(points[:, 0])])
    topmost = tuple(points[np.argmin(points[:, 1])])

    debug_info = {
        "bbox": (x, y, bw, bh),
        "centroid": (cx, cy),
        "left": leftmost,
        "right": rightmost,
        "top": topmost,
        "direction": direction,
        "horizontal_spread": horizontal_spread,
        "vertical_spread": vertical_spread,
        "left_mass": left_mass,
        "right_mass": right_mass,
    }
    return direction, debug_info


def detect_arrow(arrow_roi):
    """Detect arrow direction from the selected ROI using pixel-mass
    distribution (robust to blur / anti-aliasing / minor contour noise)."""
    binary = _binarize_arrow(arrow_roi)
    contour = _largest_arrow_contour(binary)  # kept only for debug drawing
    direction, debug_info = _classify_arrow_by_mass(binary, contour)
    debug_info["binary"] = binary
    debug_info["contour"] = contour
    return direction, debug_info


def navigation_decision(direction):
    """Map a detected arrow direction to a navigation command string."""
    if direction not in NAVIGATION_TABLE:
        raise ValueError(f"Unknown arrow direction: {direction}")
    return NAVIGATION_TABLE[direction]


def _draw_arrow_debug(arrow_roi, debug_info):
    """Draw the selected contour, bounding box and key points."""
    vis = arrow_roi.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    contour = debug_info["contour"]
    x, y, w, h = debug_info["bbox"]
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 1)
    cv2.circle(vis, debug_info["centroid"], 4, (0, 255, 255), -1)
    cv2.circle(vis, tuple(debug_info["left"]), 4, (255, 0, 0), -1)
    cv2.circle(vis, tuple(debug_info["right"]), 4, (0, 0, 255), -1)
    cv2.circle(vis, tuple(debug_info["top"]), 4, (0, 255, 0), -1)
    cv2.putText(vis, debug_info["direction"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def show_debug(windows, output_dir=None, use_gui=True):
    """
    Display (and optionally save) all debug windows.

    Args:
        windows (dict[str, ndarray]): window title -> image
        output_dir (str or None): if given, every window is also written
            to disk as a PNG in this directory.
        use_gui (bool): if True, attempt cv2.imshow (requires a display).
            If no display is available, this falls back to save-only mode
            automatically.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for title, img in windows.items():
            safe_name = title.lower().replace(" ", "_") + ".png"
            cv2.imwrite(os.path.join(output_dir, safe_name), img)
        print(f"[INFO] Debug images saved to: {os.path.abspath(output_dir)}")

    if not use_gui:
        return

    try:
        for title, img in windows.items():
            cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        print(f"[WARN] GUI display unavailable ({exc}). "
              f"Debug images were saved to disk instead." if output_dir else
              f"[WARN] GUI display unavailable ({exc}). "
              f"Re-run with --output-dir to save debug images instead.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NXP CUP India 2026 - six-panel sign board arrow detector."
    )
    parser.add_argument(
        "--goal",
        required=True,
        choices=list(GOAL_COLUMN_MAP.keys()),
        help="Target goal letter (A, B, C, X, Y, or Z).",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the sign board image.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug_output",
        help="Directory to save debug images (default: debug_output).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open cv2.imshow windows; only save debug images to disk.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        # Step 1 + 2: detect the green board.
        ordered_pts, green_mask, board_debug = detect_board(image)

        # Step 3: warp the board into a flat rectangle.
        warped = warp_board(image, ordered_pts)

        # Step 4: split into six equal columns.
        columns = split_into_columns(warped)

        # Step 5: select the column corresponding to the goal letter.
        selected_index = GOAL_COLUMN_MAP[args.goal]
        selected_column_img, col_x_offset = columns[selected_index]

        # Draw the selected column boundary on the warped board for debug.
        selected_column_debug = warped.copy()
        col_width = selected_column_img.shape[1]
        cv2.rectangle(
            selected_column_debug,
            (col_x_offset, 0),
            (col_x_offset + col_width, warped.shape[0]),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            selected_column_debug,
            f"GOAL {args.goal}",
            (col_x_offset + 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Step 6: extract only the arrow region of that column.
        arrow_roi, roi_x_offset, roi_y_offset = extract_arrow_roi(selected_column_img)

        # Arrow detection using contour geometry.
        direction, debug_info = detect_arrow(arrow_roi)
        decision = navigation_decision(direction)

        # --- Build final annotated result image -----------------------
        final_result = warped.copy()
        cv2.rectangle(
            final_result,
            (col_x_offset, 0),
            (col_x_offset + col_width, warped.shape[0]),
            (0, 0, 255),
            3,
        )
        # Translate arrow contour points from ROI-local coords into the
        # selected-column coordinate space, then into warped-board space.
        contour_global = debug_info["contour"].copy()
        contour_global[:, 0, 0] += col_x_offset + roi_x_offset
        contour_global[:, 0, 1] += roi_y_offset
        cv2.drawContours(final_result, [contour_global], -1, (0, 255, 0), 2)

        bbox_x, bbox_y, bbox_w, bbox_h = debug_info["bbox"]
        bbox_global = (
            bbox_x + col_x_offset + roi_x_offset,
            bbox_y + roi_y_offset,
            bbox_w,
            bbox_h,
        )
        cv2.rectangle(
            final_result,
            (bbox_global[0], bbox_global[1]),
            (bbox_global[0] + bbox_global[2], bbox_global[1] + bbox_global[3]),
            (255, 0, 255),
            2,
        )
        cv2.putText(
            final_result,
            f"{args.goal}: {direction}",
            (col_x_offset + 5, warped.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            final_result,
            decision,
            (10, warped.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

        arrow_debug = _draw_arrow_debug(arrow_roi, debug_info)
        binary_debug = cv2.cvtColor(debug_info["binary"], cv2.COLOR_GRAY2BGR)

        # --- Print required output --------------------------------------
        print(f"Goal : {args.goal}")
        print(f"Selected Column : {selected_index}")
        print(f"Detected Direction : {direction}")
        print(f"Navigation Decision : {decision}")

        # --- Show / save debug windows -----------------------------------
        windows = {
            "Original": image,
            "Warped Board": warped,
            "Selected Column": selected_column_debug,
            "Arrow ROI": arrow_debug,
            "Arrow Binary": binary_debug,
            "Final Result": final_result,
        }
        show_debug(windows, output_dir=args.output_dir, use_gui=not args.no_gui)

    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
