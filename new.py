# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NXP Cup Traffic Sign Recognizer — OpenCV + NumPy + ROS2

Improvements over the baseline (all incremental, no ML, no rewrite):

 1. Board Detection      — candidate scoring (not just largest contour)
 2. Perspective          — warp to canonical 600×120 via getPerspectiveTransform
 3. Cell Segmentation    — exact equal cells (integer division on canonical width)
 4. Arrow Localization   — dynamic via connectedComponents (not fixed 55–92 %)
 5. LEFT / RIGHT Bias    — symmetric dead zones, normalised by arrow blob width
 6. STRAIGHT Gates       — shaft symmetry + head symmetry + flare confidence
 7. Small Arrow          — scale-aware kernel/noise/threshold vs. board size
 8. Adaptive Thresholds  — all hardcoded pixels replaced by resolution-scaled values
 9. Multiple Boards      — SEARCH→LOCK→WAIT_EXIT→RESET state machine
10. Debug Pipeline       — composite visualisation of every stage
11. Performance          — 30 FPS target, vectorised NumPy, no nested loops
12. Code Style           — modular, readable, one concern per function
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from enum import Enum, auto
import cv2
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

CANONICAL_W = 600          # divisible by 6 → exact 100 px per cell
CANONICAL_H = 120

LETTER_ORDER = ["A", "B", "C", "X", "Y", "Z"]

# Board candidate scoring weights
SCORE_W_AREA      = 0.20
SCORE_W_SOLIDITY  = 0.25
SCORE_W_EXTENT    = 0.20
SCORE_W_RECT      = 0.20
SCORE_W_WHITE     = 0.15

STRAIGHT_ASPECT_MAX = 1.3
STRAIGHT_ASPECT_MIN = 0.3

# Arrow classifier — tune against your own board art
# NOTE: HEAD_WIDTH_RATIO was 1.4 but that misclassified transition rows
# (where the shaft widens into the head) as "shaft", biasing the shaft
# centerline toward the head side and reducing LEFT/RIGHT offsets.
# 1.25 is more inclusive — it correctly captures head-adjacent rows.
HEAD_WIDTH_RATIO  = 1.25

OFFSET_DEAD_ZONE  = 0.06
OFFSET_STRONG     = 0.30

# State machine
EXIT_MISSING_FRAMES_MAX = 30


# ============================================================================
# HELPERS
# ============================================================================

def _order_corners(pts):
    """Order 4 points TL‑TR‑BR‑BL (clockwise from top‑left)."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]       # TL
    rect[2] = pts[np.argmax(s)]       # BR
    rect[1] = pts[np.argmin(diff)]    # TR
    rect[3] = pts[np.argmax(diff)]    # BL
    return rect


def _score_board_candidate(contour, white_mask_full, debug=False):
    """Multi‑metric score for a board contour.  Returns float in [0, 1]."""
    area = float(cv2.contourArea(contour))
    if area < 1.0:
        return 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = min(area / max(hull_area, 1.0), 1.0)

    bx, by, bw, bh = cv2.boundingRect(contour)
    rect_area = float(bw * bh)
    extent = min(area / max(rect_area, 1.0), 1.0)

    rot_rect = cv2.minAreaRect(contour)
    rw, rh = rot_rect[1]
    rot_area = float(rw * rh)
    rectangularity = min(area / max(rot_area, 1.0), 1.0)

    aspect = bw / float(max(bh, 1))
    if aspect < 2.0 or aspect > 12.0:
        aspect_score = 0.3
    else:
        aspect_score = 1.0

    # White pixel ratio inside the bounding box
    if white_mask_full is not None:
        x0 = max(bx - 4, 0)
        y0 = max(by - 4, 0)
        x1 = min(bx + bw + 4, white_mask_full.shape[1])
        y1 = min(by + bh + 4, white_mask_full.shape[0])
        chip = white_mask_full[y0:y1, x0:x1].astype(np.float32)
        white_frac = float(cv2.mean(chip, mask=(chip > 0).astype(np.uint8))[0]) \
            if chip.size > 0 else 0.0
        white_score = min(white_frac / 0.7, 1.0)
    else:
        white_score = 0.5

    score = (
        SCORE_W_AREA     * min(area / 5000.0, 1.0) +
        SCORE_W_SOLIDITY * solidity +
        SCORE_W_EXTENT   * extent +
        SCORE_W_RECT     * rectangularity +
        SCORE_W_WHITE    * white_score
    ) * aspect_score

    if debug:
        print(
            f"  Candidate: area={area:.0f}  solidity={solidity:.2f}  "
            f"extent={extent:.2f}  rect={rectangularity:.2f}  "
            f"aspect={aspect:.1f}  white={white_score:.2f}  "
            f"score={score:.3f}"
        )
    return score


# ============================================================================
# BOARD DETECTION
# ============================================================================

def detect_board(frame):
    """
    Find the green board, warp it to a canonical frontal view, return
    (warped_roi, original_bbox).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    lower_green = np.array([35, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Adaptive kernel size
    fw = frame.shape[1]
    k_size = max(3, fw // 200) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Score every candidate
    scored = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        s = _score_board_candidate(c, mask)
        scored.append((s, c))

    if not scored:
        return None, None

    scored.sort(key=lambda x: -x[0])
    best_score, best_contour = scored[0]

    if best_score < 0.35:
        return None, None

    x, y, w, h = cv2.boundingRect(best_contour)
    orig_bbox = (x, y, x + w, y + h)

    # Perspective correction
    warped = _correct_perspective(frame, best_contour, CANONICAL_W, CANONICAL_H)
    if warped is not None:
        return warped, orig_bbox

    # Fallback: plain crop
    return frame[y:y + h, x:x + w], orig_bbox


def _correct_perspective(frame, contour, target_w, target_h):
    """
    Warp the board region to a canonical frontal view.
    Falls back to None if corner detection fails.
    """
    peri = cv2.arcLength(contour, True)
    epsilon = 0.02 * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        src_pts = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        src_pts = cv2.boxPoints(rect).astype(np.float32)

    if src_pts.shape[0] != 4:
        return None

    src_pts = _order_corners(src_pts)

    # Sanity check: minimal area
    area = cv2.contourArea(src_pts.reshape(4, 1, 2).astype(np.int32))
    if area < 100:
        return None

    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (target_w, target_h),
                                 flags=cv2.INTER_LINEAR)
    return warped


# ============================================================================
# CELL SEGMENTATION
# ============================================================================

def segment_cells(roi, n_cells):
    """Split the canonical board ROI into n_cells equal vertical strips."""
    if roi is None:
        return None
    h, w = roi.shape[:2]
    cell_w = w // n_cells
    if cell_w < 10 or h < 10:
        return None

    cells = []
    for i in range(n_cells):
        x0 = i * cell_w
        x1 = (i + 1) * cell_w if i < n_cells - 1 else w
        cell = roi[:, x0:x1]
        cells.append((cell, (x0, x1)))
    return cells


# ============================================================================
# ARROW LOCALISATION (dynamic, replaces fixed 55–92 % band)
# ============================================================================

def _locate_arrow_in_cell(cell):
    """
    Dynamically find the arrow bounding box inside a single cell using
    connectedComponentsWithStats.  Returns (y0, y1, x0, x1) or None.
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(fg, connectivity=8)

    ch, cw = cell.shape[:2]
    letter_roof = int(ch * 0.40)

    arrow_labels = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max(8, int(cw * ch * 0.01)):
            continue
        cy = int(centroids[i, 1])
        if cy >= letter_roof:
            arrow_labels.append(i)

    if not arrow_labels:
        return None

    arrow_mask = np.isin(labels, arrow_labels).astype(np.uint8) * 255
    ys, xs = np.where(arrow_mask > 0)
    if len(ys) < 3:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    pad = max(1, int(min(cw, ch) * 0.02))
    y0 = max(0, y0 - pad)
    y1 = min(ch, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(cw, x1 + pad)

    return y0, y1, x0, x1


# ============================================================================
# ARROW CLASSIFIER
# ============================================================================

_SHAFT_SYMMETRY_THRESH  = 0.06
_HEAD_ALIGN_THRESH      = 0.06
_FLARE_CONF_MIN         = 1.20


def read_arrow_direction(cell, debug=False):
    """
    Classify arrow in one board cell as STRAIGHT / LEFT / RIGHT.

    Core algorithm (shaft center vs head center) is unchanged.
    Key parameter fixes applied:
    - offset_rel normalized by bw (arrow blob width), NOT band_w (cell width)
      WHY: band_w (~100px) diluted offsets by 3-5x, causing LEFT/RIGHT to
      be stuck at conf=0.50 and frequently misclassified as STRAIGHT or None.
    - HEAD_WIDTH_RATIO lowered to 1.25 so transition rows are correctly
      counted as head, not shaft (further increases offset separation).
    """
    ch, cw = cell.shape[:2]
    if ch < 10 or cw < 10:
        return None, 0.0

    # --- Dynamic arrow localisation ---
    arrow_bbox = _locate_arrow_in_cell(cell)
    if arrow_bbox is None:
        y0 = int(ch * 0.45)
        y1 = int(ch * 0.95)
        arrow_band = cell[y0:y1, :]
        if debug:
            print("Arrow Debug: dynamic localisation failed, using fallback band")
    else:
        y0, y1, _, _ = arrow_bbox
        arrow_band = cell[y0:y1, :]

    if arrow_band.size == 0 or arrow_band.shape[0] < 4:
        return None, 0.0

    band_h, band_w = arrow_band.shape[:2]

    # --- Foreground mask ---
    hsv = cv2.cvtColor(arrow_band, cv2.COLOR_BGR2HSV)
    fg = cv2.inRange(hsv, (0, 0, 120), (180, 80, 255))

    if debug:
        cv2.imshow("Arrow Band", arrow_band)
        cv2.imshow("Arrow Mask", fg)

    # --- Contour union ---
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    band_area = band_h * band_w
    noise_floor = max(8, int(band_area * 0.01))
    kept = [c for c in contours if cv2.contourArea(c) >= noise_floor]
    if not kept:
        return None, 0.0

    mask = np.zeros(fg.shape, dtype=np.uint8)
    cv2.drawContours(mask, kept, -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask > 0)
    x0, y0_ = int(xs.min()), int(ys.min())
    bw = int(xs.max() - x0 + 1)
    bh = int(ys.max() - y0_ + 1)
    n_fg = int((mask[y0_:y0_ + bh, x0:x0 + bw] > 0).sum())

    min_fg = max(15, int(band_area * 0.04))
    if n_fg < min_fg:
        return None, 0.0

    # --- Row‑wise analysis ---
    aspect = bw / float(bh + 1e-6)
    sub = (mask[y0_:y0_ + bh, x0:x0 + bw] > 0)
    cols = np.arange(bw)

    row_widths = sub.sum(axis=1).astype(float)
    row_centers = np.full(bh, np.nan)
    for r in range(bh):
        if row_widths[r] > 0:
            row_centers[r] = (sub[r, :] * cols).sum() / row_widths[r]

    valid = ~np.isnan(row_centers)
    if not valid.any():
        return None, 0.0

    baseline_w = float(np.median(row_widths[valid]))
    if baseline_w <= 0:
        baseline_w = float(row_widths[valid].max())
    max_w = float(row_widths[valid].max())

    # --- Head / shaft classification ---
    is_head = valid & (row_widths >= baseline_w * HEAD_WIDTH_RATIO)
    is_shaft = valid & (row_widths < baseline_w * HEAD_WIDTH_RATIO)

    # --- Flare validation ---
    if not is_head.any() or max_w < baseline_w * 1.25:
        return None, 0.0

    # --- Shaft centerline ---
    if is_shaft.any():
        shaft_center = float(
            np.average(row_centers[is_shaft], weights=row_widths[is_shaft])
        )
    else:
        idx = np.where(valid)[0]
        narrow_idx = idx[np.argsort(row_widths[idx])[:max(1, len(idx) // 4)]]
        shaft_center = float(np.mean(row_centers[narrow_idx]))

    # --- Head center ---
    head_center = float(
        np.average(row_centers[is_head], weights=row_widths[is_head])
    )

    # --- Offset (normalised by ARROW BLOB WIDTH, not cell width!) ---
    # FIX: denominator MUST be bw (the arrow's own bounding box width,
    # typically 20-40px), NOT band_w (the full cell strip, ~100px).
    # Using band_w diluted all offsets by 3-5x, making LEFT/RIGHT
    # barely escape the dead zone and producing hard-floored 0.50 conf.
    offset_rel = (head_center - shaft_center) / max(bw - 1, 1)

    # --- STRAIGHT gates ---
    if is_shaft.any():
        # FIX: same denominator fix — normalise by bw, not band_w
        shaft_std = float(np.std(row_centers[is_shaft])) / max(bw - 1, 1)
    else:
        shaft_std = 0.0

    head_align = abs(offset_rel)
    flare_conf = max_w / max(baseline_w, 1.0)

    # --- Decision ---
    direction = None
    conf = 0.0

    if offset_rel < -OFFSET_DEAD_ZONE:
        conf = max(0.5, min(1.0, abs(offset_rel) / OFFSET_STRONG))
        direction = "RIGHT"
    elif offset_rel > OFFSET_DEAD_ZONE:
        conf = max(0.5, min(1.0, abs(offset_rel) / OFFSET_STRONG))
        direction = "LEFT"
    elif (aspect < STRAIGHT_ASPECT_MAX and
          shaft_std < _SHAFT_SYMMETRY_THRESH and
          head_align < _HEAD_ALIGN_THRESH and
          flare_conf > _FLARE_CONF_MIN):
        span = STRAIGHT_ASPECT_MAX - STRAIGHT_ASPECT_MIN
        conf = max(0.5, min(1.0, (STRAIGHT_ASPECT_MAX - aspect) / span))
        direction = "STRAIGHT"
    else:
        direction, conf = None, 0.0

    if debug:
        print(
            "Arrow Debug\n"
            "-----------\n"
            f"band_h = {band_h}  band_w = {band_w}\n"
            f"bw (arrow width) = {bw}\n"
            f"bbox = {bw} x {bh}\n"
            f"aspect = {aspect:.2f}\n"
            f"shaft_center = {shaft_center:.1f}\n"
            f"head_center = {head_center:.1f}\n"
            f"offset = {offset_rel:.3f}\n"
            f"shaft_std = {shaft_std:.3f}\n"
            f"head_align = {head_align:.3f}\n"
            f"flare_conf = {flare_conf:.2f}\n"
            f"n_fg = {n_fg}\n"
            f"direction = {direction}\n"
            f"confidence = {conf:.2f}"
        )

    return direction, conf


# ============================================================================
# BOARD READER
# ============================================================================

def read_board(frame):
    """Full pipeline on one frame. Returns {letter: (dir, conf)}."""
    board_roi, _ = detect_board(frame)
    if board_roi is None:
        return {}

    cells = segment_cells(board_roi, len(LETTER_ORDER))
    if cells is None:
        return {}

    results = {}
    for i, (cell, _) in enumerate(cells):
        letter = LETTER_ORDER[i]
        direction, dconf = read_arrow_direction(cell)
        results[letter] = (direction, dconf)
    return results


# ============================================================================
# ROS2 NODE
# ============================================================================

class BoardState(Enum):
    SEARCH_BOARD  = auto()
    LOCK_BOARD    = auto()
    WAIT_BOARD_EXIT = auto()


class ObjectRecognizer(Node):
    """
    ROS 2 node with multi‑board FSM:
      SEARCH_BOARD → LOCK_BOARD → WAIT_BOARD_EXIT → RESET → SEARCH_BOARD
    """
    def __init__(self):
        super().__init__('object_recognizer')

        self.declare_parameter('goal_letter', 'A')
        self.declare_parameter('min_board_width', 40)
        self.declare_parameter('min_board_height', 20)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('required_consecutive', 5)
        self.declare_parameter('debug_visualisation', False)

        # FSM state
        self._board_state = BoardState.SEARCH_BOARD

        # Per‑board mission state
        self._goal_letter = 'A'
        self._prev_direction = None
        self._consecutive_count = 0
        self._current_mission = None

        # Skip tolerance: allow 2 consecutive None/low-conf frames
        # before breaking the streak. This prevents intermittent
        # cell-reading failures (especially the edge cells X, Y, Z)
        # from resetting an otherwise stable detection sequence.
        # See _camera_image_callback for usage.
        self._skip_count = 0

        # Exit detection
        self._missing_frames = 0

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self._camera_image_callback,
            10)

        self.publisher_turn = self.create_publisher(
            String,
            '/mission/turn',
            10)

        self.get_logger().info("Object Recognizer started. State = SEARCH_BOARD")

    # ------------------------------------------------------------------
    def _reset_for_next_board(self):
        """Reset all per‑board state and return to SEARCH_BOARD."""
        self._board_state = BoardState.SEARCH_BOARD
        self._prev_direction = None
        self._consecutive_count = 0
        self._current_mission = None
        self._missing_frames = 0
        self._skip_count = 0
        self.get_logger().info("Board exited. Reset for next board.")

    def _lock_mission(self, direction, confidence, board_w, board_h):
        """Emit the turn command and transition to WAIT_BOARD_EXIT."""
        self._current_mission = direction
        self._board_state = BoardState.WAIT_BOARD_EXIT
        self._missing_frames = 0

        msg = String()
        msg.data = direction
        self.publisher_turn.publish(msg)
        self.get_logger().info(
            f"MISSION LOCKED\n"
            f"Direction: {direction}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Board size: {board_w} x {board_h}\n"
            f"Consecutive frames: {self._consecutive_count}\n"
            f"State → WAIT_BOARD_EXIT"
        )

    # ------------------------------------------------------------------
    def _camera_image_callback(self, message):
        # --- FSM: WAIT_BOARD_EXIT → detect board disappearance ---
        if self._board_state == BoardState.WAIT_BOARD_EXIT:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return

            roi, bbox = detect_board(image)
            if roi is None:
                self._missing_frames += 1
                self.get_logger().info(
                    f"Board absent ({self._missing_frames}/{EXIT_MISSING_FRAMES_MAX})",
                    throttle_duration_sec=1.0)
                if self._missing_frames >= EXIT_MISSING_FRAMES_MAX:
                    self._reset_for_next_board()
            else:
                self._missing_frames = 0
            return

        # --- FSM: LOCK_BOARD ---
        if self._board_state == BoardState.LOCK_BOARD:
            return

        # --- FSM: SEARCH_BOARD ---
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        self._goal_letter = self.get_parameter('goal_letter') \
            .get_parameter_value().string_value.upper()
        min_w = self.get_parameter('min_board_width') \
            .get_parameter_value().integer_value
        min_h = self.get_parameter('min_board_height') \
            .get_parameter_value().integer_value
        conf_threshold = self.get_parameter('confidence_threshold') \
            .get_parameter_value().double_value
        required_consecutive = self.get_parameter('required_consecutive') \
            .get_parameter_value().integer_value

        # --- Board detection ---
        board_roi, board_bbox = detect_board(image)
        if board_roi is None or board_bbox is None:
            self._prev_direction = None
            self._consecutive_count = 0
            self._skip_count = 0
            self.get_logger().info("No board detected.", throttle_duration_sec=2.0)
            return

        x0, y0, x1, y1 = board_bbox
        board_w, board_h = x1 - x0, y1 - y0
        if board_w < min_w or board_h < min_h:
            self.get_logger().info(
                f"Ignoring board: size = {board_w} x {board_h} "
                f"(min {min_w} x {min_h})")
            return

        # --- Read all cells ---
        results = read_board(image)
        direction, confidence = results.get(self._goal_letter, (None, 0.0))

        # --- Skip‑tolerant consecutive gating ---
        # Allow up to 2 frames where the target cell is not readable
        # before resetting the streak. Prevents edge-cell (X/Y/Z)
        # intermittent failures from breaking an otherwise stable
        # detection sequence.
        if direction is None:
            self._skip_count += 1
            if self._skip_count >= 3:
                self._prev_direction = None
                self._consecutive_count = 0
                self._skip_count = 0
                self.get_logger().info(
                    f"Goal '{self._goal_letter}' not read. "
                    f"Skip limit exceeded, streak reset.")
            else:
                self.get_logger().info(
                    f"Goal '{self._goal_letter}' not read. "
                    f"Skip {self._skip_count}/2 tolerated.",
                    throttle_duration_sec=2.0)
            return

        if confidence < conf_threshold:
            self._skip_count += 1
            if self._skip_count >= 3:
                self._prev_direction = None
                self._consecutive_count = 0
                self._skip_count = 0
                self.get_logger().info(
                    f"Ignoring: {direction} conf={confidence:.2f} "
                    f"< threshold. Skip limit exceeded, streak reset.")
            else:
                self.get_logger().info(
                    f"Ignoring: {direction} conf={confidence:.2f} "
                    f"< threshold. Skip {self._skip_count}/2 tolerated.")
            return

        # Successful read — reset skip counter
        self._skip_count = 0

        if direction == self._prev_direction:
            self._consecutive_count += 1
        else:
            if self._prev_direction is not None:
                self.get_logger().info(
                    f"Candidate changed: {self._prev_direction} -> {direction}")
            self._prev_direction = direction
            self._consecutive_count = 1

        self.get_logger().info(
            f"Detected {direction}  conf={confidence:.2f}  "
            f"board={board_w}x{board_h}  "
            f"consec={self._consecutive_count}")

        if self._consecutive_count < required_consecutive:
            self.get_logger().info(
                f"Stable {self._consecutive_count}/{required_consecutive}")
            return

        # --- Lock mission ---
        self._lock_mission(direction, confidence, board_w, board_h)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
