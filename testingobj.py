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

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

try:
    from synapse_msgs.msg import ServerCommunication
except ImportError:
    ServerCommunication = None

try:
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    get_message = None

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

# ============================================================================
# FAST CLASSICAL-CV SIGN DETECTOR
# ============================================================================
# No ML / DL / OCR.
# Pipeline per board:
#   green board detection -> perspective warp -> equal 6-cell split ->
#   dual-window white arrow extraction -> geometry classification.
#
# Multi-board support is kept: every detected mid-range board is processed.
# The ROS node chooses the highest-confidence result for the current goal.
# ============================================================================

CANONICAL_W = 600
CANONICAL_H = 120
LETTER_ORDER = ["A", "B", "C", "X", "Y", "Z"]
CELL_LABELS = LETTER_ORDER

# Mid-range gate. Increase these if it still detects too early/far away.
MIN_DETECT_BOARD_WIDTH = 180
MIN_DETECT_BOARD_HEIGHT = 48
MIN_BOARD_AREA = 1200

# When multiple boards are visible, do not choose the highest-confidence arrow
# from any board in the image. That can pick a side/old board. Choose the board
# that is in front of the robot: near image center and reasonably large.
BOARD_CENTER_GATE = 0.65       # 0=center, 1=edge. Reject boards beyond this.
BOARD_CENTER_WEIGHT = 2.0      # higher = stronger preference for centered board

# Board exit reset. After mission lock, the node waits until the board is gone
# for this many frames, then unlocks and searches again for the SAME current
# goal. /ServerCommunication can change the goal at any time.
EXIT_MISSING_FRAMES_MAX = 12

# Do not tolerate missed/low-confidence frames for locking. A single bad frame
# resets the streak, preventing false RIGHT/STRAIGHT locks while approaching.
SKIP_MISSING_FRAMES_MAX = 0

GREEN_HSV_LOW = np.array([35, 35, 35])
GREEN_HSV_HIGH = np.array([95, 255, 255])

# Arrow extraction / classification tuning.
#
# These signs place the arrow at two different vertical positions:
#   * lower cell       -> ARROW_Y0_RATIO .. ARROW_Y1_RATIO
#   * upper-middle cell -> ARROW_UPPER_Y0 .. ARROW_UPPER_Y1
# read_arrow_direction() tries BOTH windows and keeps the confident read. A
# single fixed window matched only one layout (39%); the dual window hits both.
ARROW_Y0_RATIO = 0.58
ARROW_Y1_RATIO = 0.96
ARROW_UPPER_Y0 = 0.30
ARROW_UPPER_Y1 = 0.55
ARROW_X_MARGIN_RATIO = 0.07
WHITE_HSV_LOW = np.array([0, 0, 115])
WHITE_HSV_HIGH = np.array([180, 125, 255])

# A single arrow cannot fill most of the ROI. If the selected white blob covers
# more than this fraction of the ROI, the cell is contaminated (white margin /
# glare / mis-warp) and its direction is unreliable -> return None instead of a
# flippy low-confidence call.
ARROW_FILL_MAX = 0.45

# Straight must be REALLY vertical/narrow. Earlier value 1.45 caused broken
# LEFT/RIGHT arrow fragments to be called STRAIGHT at mid range.
STRAIGHT_ASPECT_MAX = 1.05
# Treat moderate-width blobs as horizontal arrows; if centroid is weak they
# will be skipped instead of guessed.
HORIZONTAL_ASPECT_MIN = 1.20
DIR_ASYM_DEAD_ZONE = 0.10
DIR_ASYM_STRONG = 0.25
# Horizontal LEFT/RIGHT decision. Centroid is more stable than row-extents at
# mid/far range. Negative centroid offset = LEFT, positive = RIGHT.
CENTROID_DEAD_ZONE = 0.025
CENTROID_STRONG = 0.055
# Do not accept weak STRAIGHT reads. False A/B straight mistakes were
# low-confidence (~0.82). Real straight arrows in samples score >0.90, so weak
# STRAIGHT reads are ignored.
STRAIGHT_ACCEPT_CONF = 0.90
DEFAULT_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_REQUIRED_CONSECUTIVE = 5


def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def get_board_quad(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    return order_points(quad)


def correct_perspective(frame, quad, target_w=CANONICAL_W, target_h=CANONICAL_H):
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, matrix, (target_w, target_h), flags=cv2.INTER_LINEAR)


def green_board_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_boards(frame):
    """Return all mid-range green boards, sorted largest first."""
    mask = green_board_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boards = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BOARD_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_DETECT_BOARD_WIDTH or h < MIN_DETECT_BOARD_HEIGHT:
            continue
        aspect = w / float(h + 1e-6)
        if aspect < 2.0:
            continue
        quad = get_board_quad(contour)
        warped = correct_perspective(frame, quad)
        boards.append({
            "contour": contour,
            "quad": quad,
            "warped": warped,
            "bbox": (x, y, x + w, y + h),
            "area": area,
        })
    boards.sort(key=lambda b: -b["area"])
    return boards


def detect_board(frame):
    """Compatibility helper: returns the largest board ROI and bbox."""
    boards = detect_boards(frame)
    if not boards:
        return None, None
    board = boards[0]
    return board["warped"], board["bbox"]


def split_cells(warped, n_cells=6):
    h, w = warped.shape[:2]
    cell_w = w // n_cells
    cells = []
    for i in range(n_cells):
        x0 = i * cell_w
        x1 = (i + 1) * cell_w if i < n_cells - 1 else w
        cells.append((warped[:, x0:x1], (x0, x1)))
    return cells


def arrow_mask_from_cell(cell, y0_ratio=ARROW_Y0_RATIO, y1_ratio=ARROW_Y1_RATIO):
    ch, cw = cell.shape[:2]
    y0 = int(ch * y0_ratio)
    y1 = int(ch * y1_ratio)
    xm = max(2, int(cw * ARROW_X_MARGIN_RATIO))
    roi = cell[y0:y1, xm:cw - xm]
    if roi.size == 0:
        return None, None, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask, roi, (xm, y0)


def select_arrow_component(mask):
    """Return a cleaned UNION of arrow components, not just the largest blob.

    Straight/up arrows can split into two pieces at mid range: triangular head
    and vertical shaft. Picking only the largest piece makes B/Z look like a
    horizontal LEFT/RIGHT arrow. So we keep all reasonable arrow pieces and only
    reject tiny noise and edge divider artifacts.
    """
    h, w = mask.shape[:2]
    roi_area = h * w
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    selected = np.zeros_like(mask)
    min_area = max(10, int(roi_area * 0.003))
    kept_any = False
    best_score = 0.0
    best_id = None
    for cid in range(1, n_labels):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        x, y, bw, bh = stats[cid, 0:4]
        cx, cy = centroids[cid]
        if area < min_area:
            continue
        # Reject tiny specks.
        if bw < 3 or bh < 3:
            continue
        near_edge = (x <= 2) or (x + bw >= w - 2)
        # Reject small edge specks/fragments that expand the bbox and ruin
        # STRAIGHT detection, especially in Z.
        if near_edge and area < 45:
            continue
        # Reject vertical divider fragments. Dividers are narrow and almost
        # full ROI height. A true STRAIGHT shaft is narrow too, but it is not
        # full height and is usually connected to the arrow head.
        if bw <= 10 and bh > 0.72 * h:
            continue
        if near_edge and bw <= 8 and bh > 0.30 * h:
            continue
        # Reject high fragments; after the lower crop, real arrow pixels are
        # still in the middle/lower part of the ROI. This prevents letter
        # pieces, especially the vertical part of 'A', from being classified
        # as a STRAIGHT arrow at mid range.
        if cy < 0.18 * h:
            continue
        selected[labels == cid] = 255
        kept_any = True
        bbox_area = float(bw * bh)
        extent = area / max(bbox_area, 1.0)
        cy_bonus = 0.5 + cy / float(h + 1e-6)
        score = area * extent * cy_bonus
        if score > best_score:
            best_score = score
            best_id = cid
    if kept_any:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel, iterations=1)
        return selected
    # Fallback: keep best component if everything was filtered too hard.
    if best_id is not None:
        selected[labels == best_id] = 255
        return selected
    return None


def classify_arrow_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size < 20 or ys.size < 20:
        return None, 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    if bw < 7 or bh < 7:
        return None, 0.0
    aspect = bw / float(bh + 1e-6)
    pixels = int(xs.size)
    sub = (mask[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8)
    col_counts = sub.sum(axis=0).astype(np.float32)
    row_counts = sub.sum(axis=1).astype(np.float32)
    max_col_fill = float(col_counts.max()) / float(max(bh, 1))
    max_row_fill = float(row_counts.max()) / float(max(bw, 1))

    # ------------------------------------------------------------------
    # STRAIGHT / UP detector.
    # A straight arrow has a central vertical shaft + head. The important
    # signature is a tall vertical column. This is checked BEFORE left/right
    # so B/Z do not become LEFT/RIGHT when head/shaft split or blur.
    # ------------------------------------------------------------------
    vertical_like = (
        pixels >= 35 and
        bh >= 14 and
        aspect <= 1.35 and
        max_col_fill >= 0.45
    )
    very_vertical = (
        pixels >= 35 and
        bh >= 18 and
        aspect <= 1.10
    )
    if vertical_like or very_vertical:
        # confidence from vertical strength and narrowness
        aspect_conf = np.clip((1.35 - aspect) / max(1.35 - 0.55, 1e-6), 0.0, 1.0)
        col_conf = np.clip(max_col_fill / 0.75, 0.0, 1.0)
        conf = max(0.82, min(1.0, 0.55 * aspect_conf + 0.45 * col_conf))
        if conf < STRAIGHT_ACCEPT_CONF:
            return None, 0.0
        return "STRAIGHT", float(conf)

    # LEFT/RIGHT requires a clearly horizontal arrow. If it is not clearly
    # vertical and not clearly horizontal, skip instead of guessing.
    horizontal_like = (
        aspect >= HORIZONTAL_ASPECT_MIN and
        max_row_fill >= 0.35 and
        bw >= 14
    )
    if not horizontal_like:
        return None, 0.0
    # Centroid relative to bbox center is stable for left/right on this board.
    bbox_center_x = 0.5 * (x0 + x1)
    centroid_x = float(xs.mean())
    centroid_offset = (centroid_x - bbox_center_x) / float(max(bw, 1))
    if centroid_offset < -CENTROID_DEAD_ZONE:
        direction = "LEFT"
    elif centroid_offset > CENTROID_DEAD_ZONE:
        direction = "RIGHT"
    else:
        return None, 0.0
    conf = max(0.82, min(1.0, abs(centroid_offset) / CENTROID_STRONG))
    return direction, float(conf)


def read_arrow_direction(cell, debug=False):
    """Read one cell's arrow, trying both vertical layouts and keeping the
    most confident classification. Contaminated (flooded) ROIs are rejected.

    Returns (direction, confidence); direction is None if no reliable read.
    """
    best = (None, 0.0)
    best_artifacts = None
    for y0_ratio, y1_ratio in (
        (ARROW_UPPER_Y0, ARROW_UPPER_Y1),   # upper-middle layout
        (ARROW_Y0_RATIO, ARROW_Y1_RATIO),   # lower layout
    ):
        mask, roi, offset = arrow_mask_from_cell(cell, y0_ratio, y1_ratio)
        if mask is None:
            continue
        arrow = select_arrow_component(mask)
        if arrow is None:
            continue
        # Flood-guard: one arrow cannot fill most of the ROI -> contaminated.
        if arrow.size and (arrow.sum() / 255.0) / arrow.size > ARROW_FILL_MAX:
            continue
        direction, confidence = classify_arrow_mask(arrow)
        if direction is not None and confidence > best[1]:
            best = (direction, confidence)
            best_artifacts = (roi, mask, arrow)
    if debug and best_artifacts is not None:
        roi, mask, arrow = best_artifacts
        cv2.imshow("arrow_roi", roi)
        cv2.imshow("arrow_mask", mask)
        cv2.imshow("arrow_selected", arrow)
        cv2.waitKey(1)
    return best


def read_single_board(warped):
    """Read one already-warped board: {letter: (direction, confidence)}."""
    cells = split_cells(warped, len(LETTER_ORDER))
    results = {}
    for i, (cell, _) in enumerate(cells):
        letter = LETTER_ORDER[i]
        direction, confidence = read_arrow_direction(cell)
        results[letter] = (direction, confidence)
    return results


def read_boards(frame):
    """
    Process every detected board.
    Returns a list:
        [{"bbox": (...), "warped": image, "results": {letter: (dir, conf)}}]
    """
    boards = detect_boards(frame)
    output = []
    for board in boards:
        output.append({
            "bbox": board["bbox"],
            "warped": board["warped"],
            "area": board["area"],
            "results": read_single_board(board["warped"]),
        })
    return output


def read_board(frame):
    """
    Compatibility helper.
    Returns {letter: (direction, confidence)} using the highest confidence
    across all visible boards.
    """
    board_reads = read_boards(frame)
    results = {}
    for board in board_reads:
        for letter, result in board["results"].items():
            direction, confidence = result
            prev = results.get(letter)
            if prev is None or confidence > prev[1]:
                results[letter] = result
    return results


def board_target_score(board, frame_shape):
    """Score board for mission decision. Prefer centered forward board."""
    img_h, img_w = frame_shape[:2]
    x0, y0, x1, y1 = board["bbox"]
    bw = x1 - x0
    bh = y1 - y0
    cx = 0.5 * (x0 + x1)
    center_norm = abs(cx - 0.5 * img_w) / max(0.5 * img_w, 1.0)
    if center_norm > BOARD_CENTER_GATE:
        return -1.0
    center_score = max(0.0, 1.0 - center_norm) ** BOARD_CENTER_WEIGHT
    size_score = float(bw * bh)
    return size_score * center_score


def select_target_board(board_reads, frame_shape):
    """Select the single board that is most likely ahead of the robot."""
    best = None
    best_score = -1.0
    for board in board_reads:
        score = board_target_score(board, frame_shape)
        if score > best_score:
            best_score = score
            best = board
    if best_score < 0.0:
        return None
    return best


# ============================================================================
# ROS2 NODE
# ============================================================================
class ObjectRecognizer(Node):
    """
    ROS2 node:
      - subscribes camera image
      - subscribes /ServerCommunication for the goal letter
      - processes every detected board
      - publishes locked turn direction on /mission/turn
      - after the board disappears, resets and searches again for current goal
    """

    def __init__(self):
        super().__init__('object_recognizer')

        self.declare_parameter('goal_letter', 'A')
        self.declare_parameter('min_board_width', MIN_DETECT_BOARD_WIDTH)
        self.declare_parameter('min_board_height', MIN_DETECT_BOARD_HEIGHT)
        self.declare_parameter('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        self.declare_parameter('required_consecutive', DEFAULT_REQUIRED_CONSECUTIVE)
        self.declare_parameter('exit_missing_frames', EXIT_MISSING_FRAMES_MAX)

        initial_goal = self.get_parameter('goal_letter').get_parameter_value().string_value.upper()
        self.goal_letter = initial_goal if initial_goal in LETTER_ORDER else 'A'
        self.goal_valid = True
        self._last_param_goal = self.goal_letter

        self.prev_direction = None
        self.consecutive_count = 0
        self.skip_count = 0
        self.mission_locked = False
        self.current_mission = None
        self.missing_frames = 0

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.subscription_server_communication = None
        self._server_comm_dynamic_timer = None

        if not self._create_server_communication_subscription():
            self.get_logger().warn(
                "Waiting for /ServerCommunication topic type... Will retry.")
            self._server_comm_dynamic_timer = self.create_timer(
                1.0, self._try_create_server_communication_subscription)

        self.publisher_turn = self.create_publisher(
            String,
            '/mission/turn',
            10)

        self.get_logger().info(
            f"Object Recognizer started. Multi-board enabled. Active goal = {self.goal_letter}. /ServerCommunication may change it.")

    def _extract_goal_letter(self, text):
        """Extract a valid goal letter from the text field only.

        This avoids false parsing from words like "okay".
        Examples accepted: "x", "X OKAY RIGHT", "goal:x".
        """
        if text is None:
            return None
        text = str(text).strip().upper()
        if not text:
            return None
        # First token is the cleanest format: msg: x okay right
        first = text.replace(':', ' ').replace(',', ' ').split()[0]
        if first in LETTER_ORDER:
            return first
        # Accept explicit goal=<letter> / goal:<letter> / letter=<letter>.
        normalized = text.replace(':', ' ').replace('=', ' ').replace(',', ' ')
        tokens = normalized.split()
        for i, tok in enumerate(tokens[:-1]):
            if tok in ('GOAL', 'LETTER', 'TARGET') and tokens[i + 1] in LETTER_ORDER:
                return tokens[i + 1]
        return None

    def _goal_from_server_message(self, message):
        """Map /ServerCommunication destination to goal letter.

        Mapping requested:
            dest 1 -> A
            dest 2 -> B
            dest 3 -> C
            dest 4 -> X
            dest 5 -> Y
            dest 6 -> Z

        The free-text msg field is intentionally ignored here. Your server
        message may contain words like "right", but that is NOT the goal.
        """
        dest_map = {1: 'A', 2: 'B', 3: 'C', 4: 'X', 5: 'Y', 6: 'Z'}
        try:
            dest = int(getattr(message, 'dest'))
        except Exception:
            return None
        return dest_map.get(dest)

    def _create_server_communication_subscription(self):
        """Create /ServerCommunication subscription.

        Uses synapse_msgs.msg.ServerCommunication if available. If the exact
        class name is different, it falls back to ROS2 topic introspection.
        """
        if self.subscription_server_communication is not None:
            return True
        msg_type = None
        # Prefer the real runtime topic type. This avoids hard-coding the
        # custom message class name.
        if get_message is not None:
            for topic_name, topic_types in self.get_topic_names_and_types():
                if topic_name == '/ServerCommunication' and topic_types:
                    try:
                        msg_type = get_message(topic_types[0])
                        break
                    except Exception as exc:
                        self.get_logger().warn(
                            f"Could not load /ServerCommunication type {topic_types[0]}: {exc}")
        # Fallback if the topic has not appeared yet but the generated message
        # class is available as synapse_msgs.msg.ServerCommunication.
        if msg_type is None:
            msg_type = ServerCommunication
        if msg_type is None:
            return False
        self.subscription_server_communication = self.create_subscription(
            msg_type,
            '/ServerCommunication',
            self.server_communication_callback,
            10)
        self.get_logger().info(
            f"Subscribed to /ServerCommunication ({msg_type.__module__}.{msg_type.__name__})")
        return True

    def _try_create_server_communication_subscription(self):
        if self._create_server_communication_subscription():
            if self._server_comm_dynamic_timer is not None:
                self._server_comm_dynamic_timer.cancel()
                self._server_comm_dynamic_timer = None

    def _reset_detection_state(self):
        self.prev_direction = None
        self.consecutive_count = 0
        self.skip_count = 0
        self.current_mission = None
        self.missing_frames = 0

    def _set_goal(self, new_goal, source):
        if new_goal is None or new_goal not in LETTER_ORDER:
            return False
        if new_goal != self.goal_letter or not self.goal_valid:
            self.goal_letter = new_goal
            self.goal_valid = True
            self.mission_locked = False
            self._reset_detection_state()
            self.get_logger().info(
                f"Goal set to '{new_goal}' from {source}. Detector reset.")
            return True
        return False

    def _reset_after_board_exit(self):
        """After completing one board, keep same goal and search again."""
        current_goal = self.goal_letter
        self.goal_valid = True
        self.mission_locked = False
        self._reset_detection_state()
        self.get_logger().info(
            f"Board exited. Continuing with current goal '{current_goal}'. Waiting for next board or /ServerCommunication update.")

    def server_communication_callback(self, message):
        new_goal = self._goal_from_server_message(message)
        if new_goal is None:
            self.get_logger().warn(
                f"Ignoring /ServerCommunication message without valid goal. dest={getattr(message, 'dest', None)} msg='{getattr(message, 'msg', None)}'")
            return
        self.get_logger().info(
            f"/ServerCommunication received: src={getattr(message, 'src', None)} "
            f"dest={getattr(message, 'dest', None)} "
            f"msg='{getattr(message, 'msg', None)}' -> goal={new_goal}")
        changed = self._set_goal(new_goal, '/ServerCommunication')
        # Keep parameter server synchronized with server goal.
        if changed:
            self._last_param_goal = new_goal
            self.set_parameters([
                Parameter('goal_letter', Parameter.Type.STRING, new_goal)
            ])

    def _check_parameter_goal(self):
        """
        Parameter support is kept, but it only reacts to actual parameter
        changes. The current goal persists after board exit unless
        ServerCommunication or parameter explicitly changes it.
        """
        param_goal = self.get_parameter('goal_letter').get_parameter_value().string_value.upper()
        if param_goal not in LETTER_ORDER:
            return
        if param_goal != self._last_param_goal:
            self._last_param_goal = param_goal
            self._set_goal(param_goal, 'parameter')

    def _handle_locked_wait_for_exit(self, image):
        boards = detect_boards(image)
        exit_missing_frames = self.get_parameter('exit_missing_frames').get_parameter_value().integer_value
        if not boards:
            self.missing_frames += 1
            self.get_logger().info(
                f"Board absent after mission lock ({self.missing_frames}/{exit_missing_frames})",
                throttle_duration_sec=1.0)
            if self.missing_frames >= exit_missing_frames:
                self._reset_after_board_exit()
        else:
            self.missing_frames = 0

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        # Still allow manual parameter goal changes.
        self._check_parameter_goal()

        # Mission is locked: do not publish again. Only wait for board exit.
        if self.mission_locked:
            self._handle_locked_wait_for_exit(image)
            return

        if not self.goal_valid or self.goal_letter is None:
            self.goal_letter = 'A'
            self.goal_valid = True
            self.get_logger().warn(
                "No active goal was set. Falling back to goal 'A'.")

        min_w = self.get_parameter('min_board_width').get_parameter_value().integer_value
        min_h = self.get_parameter('min_board_height').get_parameter_value().integer_value
        conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        required_consecutive = self.get_parameter('required_consecutive').get_parameter_value().integer_value

        board_reads = read_boards(image)

        if not board_reads:
            self.prev_direction = None
            self.consecutive_count = 0
            self.skip_count = 0
            self.get_logger().info("No mid-range board detected.", throttle_duration_sec=2.0)
            return

        # Multi-board detection is still enabled, but mission decision uses only
        # the forward/center board. This avoids a side board with high confidence
        # changing the mission.
        target_board = select_target_board(board_reads, image.shape)

        best = None
        if target_board is not None:
            x0, y0, x1, y1 = target_board["bbox"]
            board_w = x1 - x0
            board_h = y1 - y0
            if board_w >= min_w and board_h >= min_h:
                direction, confidence = target_board["results"].get(self.goal_letter, (None, 0.0))
                if direction is not None:
                    best = {
                        "direction": direction,
                        "confidence": confidence,
                        "bbox": target_board["bbox"],
                        "board_w": board_w,
                        "board_h": board_h,
                        "boards_seen": len(board_reads),
                    }

        if best is None:
            self.skip_count += 1
            if self.skip_count > SKIP_MISSING_FRAMES_MAX:
                self.prev_direction = None
                self.consecutive_count = 0
                self.skip_count = 0
                self.get_logger().info(
                    f"Goal '{self.goal_letter}' not read on any visible board. Streak reset.",
                    throttle_duration_sec=1.0)
            else:
                self.get_logger().info(
                    f"Goal '{self.goal_letter}' not read on this frame. Keeping streak {self.consecutive_count}; skip {self.skip_count}/{SKIP_MISSING_FRAMES_MAX}.",
                    throttle_duration_sec=1.0)
            return

        direction = best["direction"]
        confidence = best["confidence"]
        board_w = best["board_w"]
        board_h = best["board_h"]

        if confidence < conf_threshold:
            self.skip_count += 1
            if self.skip_count > SKIP_MISSING_FRAMES_MAX:
                self.prev_direction = None
                self.consecutive_count = 0
                self.skip_count = 0
                self.get_logger().info(
                    f"Ignoring {direction}: confidence = {confidence:.2f}, threshold = {conf_threshold:.2f}. Streak reset.",
                    throttle_duration_sec=1.0)
            else:
                self.get_logger().info(
                    f"Ignoring {direction}: confidence = {confidence:.2f}, threshold = {conf_threshold:.2f}. Keeping streak {self.consecutive_count}; skip {self.skip_count}/{SKIP_MISSING_FRAMES_MAX}.",
                    throttle_duration_sec=1.0)
            return

        self.skip_count = 0
        if direction == self.prev_direction:
            self.consecutive_count += 1
        else:
            if self.prev_direction is not None:
                self.get_logger().info(
                    f"Candidate changed: {self.prev_direction} -> {direction}")
            self.prev_direction = direction
            self.consecutive_count = 1

        boards_seen = best.get("boards_seen", len(board_reads))
        self.get_logger().info(
            f"Detected {direction}  goal={self.goal_letter}  conf={confidence:.2f}  "
            f"board={board_w}x{board_h}  boards_seen={boards_seen}  "
            f"consecutive={self.consecutive_count}/{required_consecutive}")

        if self.consecutive_count < required_consecutive:
            return

        self.current_mission = direction
        self.mission_locked = True
        self.missing_frames = 0

        msg = String()
        msg.data = direction
        self.publisher_turn.publish(msg)

        self.get_logger().info(
            f"MISSION LOCKED\n"
            f"Goal: {self.goal_letter}\n"
            f"Direction: {direction}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Board size: {board_w} x {board_h}\n"
            f"Consecutive frames: {self.consecutive_count}\n"
            f"State: WAIT_BOARD_EXIT")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
