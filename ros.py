#!/usr/bin/env python3
"""
detect.py

NXP CUP India 2026 - Autonomous Medical Response Challenge
ROS2 node: b3rb_ros_line_follower / detect

Everything lives inside ONE class (DetectNode) inside this ONE file, as
required. The OpenCV board / arrow detection algorithm below is the exact
same algorithm from the standalone detector.py prototype -- every OpenCV
call (thresholds, morphology, contour logic, mass-distribution
classification) is unchanged. It has simply been moved from free functions
into methods on the node class so it can run inside a ROS2 callback instead
of on a single static image loaded from disk.

Node responsibilities:
    1. Subscribe to /camera/image_raw (sensor_msgs/Image) via CvBridge.
    2. Run the existing OpenCV pipeline on every frame to read the full
       six-panel board -> self.current_board (dict: letter -> direction).
    3. Subscribe to /mission/goal (std_msgs/String) -> self.goal.
    4. If self.goal is a key in self.current_board, publish the
       corresponding direction ("LEFT" / "RIGHT" / "STRAIGHT") to
       /mission/turn (std_msgs/String).
    5. Change-only publish gate: do not republish while nothing has
       changed. Publish only when the mission goal changes, the detected
       board changes, or the board disappears and then reappears. There
       is no cooldown timer -- a previously published turn simply stays
       valid until something actually changes (see _handle_valid_board).
    6. Stability gates: every letter's raw per-frame classification is
       first checked against its own LETTER_HISTORY_LENGTH-frame history
       (see _stable_letter_vote), and only counts toward this frame's
       board if one direction holds a STRICT majority there -- an
       oscillating letter is never resolved by a guess/tie-break; it is
       simply left out, which makes the board incomplete. On top of that,
       a complete board must still be read identically for
       STABILITY_FRAMES_REQUIRED consecutive frames before it is accepted
       into self.current_board at all (see _update_pending_streak).

The class is deliberately organized into clearly separated method groups
(ROS2 plumbing / OpenCV board pipeline / mission logic) so it can be
extended later in-place with QR detection, municipality-server calls,
hospital verification, parking, and a full mission state machine, all as
additional methods on the same DetectNode class.
"""

import os
from collections import Counter, deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError


# ---------------------------------------------------------------------
# FIX 4: Debug mode.
#
# DEBUG is a plain module-level flag (not ROS-parameterized on purpose --
# it is a developer tuning switch, flipped by hand while working on the
# detector, not a runtime mission setting). When False every debug call
# site below is a single cheap `if DEBUG:` check that short-circuits
# immediately, so there is zero meaningful performance penalty on the
# hot per-frame path. When True, the warped board, each column's arrow
# ROI, and each column's binary mask (with the classification bounding
# box drawn on it) are written to DEBUG_DIR for offline tuning.
# ---------------------------------------------------------------------
DEBUG = True
DEBUG_DIR = "/tmp/detect_debug"


class DetectNode(Node):
    # -----------------------------------------------------------------
    # Class-level constants (unchanged from the standalone detector.py)
    # -----------------------------------------------------------------

    GOAL_COLUMN_MAP = {
        "A": 0,
        "B": 1,
        "C": 2,
        "X": 3,
        "Y": 4,
        "Z": 5,
    }
    COLUMN_LETTER_MAP = {v: k for k, v in GOAL_COLUMN_MAP.items()}

    NUM_COLUMNS = 6

    WARPED_WIDTH = 1200
    WARPED_HEIGHT = 400

    GREEN_HSV_LOWER = np.array([35, 40, 40])
    GREEN_HSV_UPPER = np.array([90, 255, 255])

    LETTER_REGION_FRACTION = 0.40

    NAVIGATION_TABLE = {
        "LEFT": "TURN LEFT",
        "RIGHT": "TURN RIGHT",
        "STRAIGHT": "GO STRAIGHT",
    }

    # A freshly-read complete board must repeat, identically, for this many
    # consecutive frames before it is accepted into current_board. This is
    # pure integration-layer temporal filtering -- it never touches the
    # OpenCV pipeline -- and exists to reject single-frame flukes (motion
    # blur, glare, a half-passed transition) from ever reaching navigation.
    STABILITY_FRAMES_REQUIRED = 3

    # Minimum seconds between repeats of the same throttled warning, so a
    # persistently unreadable board doesn't spam the log every frame.
    LOG_THROTTLE_SECONDS = 2.0

    # Per-letter decision stabilization. Each letter keeps its own short
    # rolling history of raw single-frame classifications. A letter only
    # contributes to board_this_frame once its history is full AND one
    # direction holds a strict majority there (see _stable_letter_vote) --
    # an oscillating letter is never guessed at, it is simply omitted.
    # This check runs independently of, and *before*, the existing
    # whole-board STABILITY_FRAMES_REQUIRED repeat-check.
    #
    # This and STABILITY_FRAMES_REQUIRED below are the ONLY two temporal
    # tuning constants in this file.
    LETTER_HISTORY_LENGTH = 5

    # ===================================================================
    # ROS2 plumbing
    # ===================================================================

    def __init__(self):
        super().__init__("detect")

        self.bridge = CvBridge()

        # ---- Mission state (no globals -- all state lives on self) ----
        # current_board starts EMPTY. It is only ever populated from a real,
        # complete (all six columns) detection -- never seeded with
        # placeholder values -- so the robot can never publish a turn
        # before it has actually seen the board.
        self.current_board = {}
        self.board_valid = False  # True once a complete board has been read
        self.goal = None

        # ---- Change-detection state (used by the change-only publish gate) ----
        # board_present_prev tracks whether the LAST FRAME produced a
        # complete, valid board (used to detect "disappeared then
        # reappeared"). last_board_signature tracks the content of the
        # last valid board seen (used to detect "board changed"), and is
        # updated on every valid detection, independent of whether a
        # publish actually happened (publishing also requires a goal).
        self.board_present_prev = False
        self.last_board_signature = None

        # ---- Per-letter decision history (FIX 2) ----
        # One fixed-length deque of raw single-frame classifications per
        # letter, independent of every other letter's history.
        self.letter_history = {
            letter: deque(maxlen=self.LETTER_HISTORY_LENGTH)
            for letter in self.GOAL_COLUMN_MAP
        }

        # ---- Temporal-filtering state (requirement 6) ----
        # A candidate board must be read identically for STABILITY_FRAMES_
        # REQUIRED consecutive frames before it replaces current_board.
        # pending_signature/pending_streak track the run currently in
        # progress; they are reset to zero the instant a differently-read
        # (or incomplete/absent) board breaks the streak.
        self.pending_signature = None
        self.pending_streak = 0

        # ---- Subscriptions ----
        # Camera images use the standard "sensor data" QoS (best-effort,
        # small history depth): frame processing should keep up with the
        # freshest frame rather than block on guaranteed delivery of every
        # historical one.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            sensor_qos,
        )
        # Mission goals are discrete commands, not a data stream -- reliable
        # delivery with a small queue is the right fit here, so this stays
        # on the default reliable QoS.
        self.goal_sub = self.create_subscription(
            String,
            "/mission/goal",
            self.goal_callback,
            10,
        )

        # ---- Publisher ----
        self.turn_pub = self.create_publisher(String, "/mission/turn", 10)

        self.get_logger().info("detect node started: waiting for camera frames.")

    # -------------------------------------------------------------
    # Subscription callbacks
    # -------------------------------------------------------------

    def image_callback(self, msg):
        """Runs the OpenCV board pipeline on every incoming camera frame."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"CvBridge conversion failed: {exc}")
            return

        self.process_frame(cv_image)

    def goal_callback(self, msg):
        """Stores the current mission goal letter and immediately checks
        whether we can already publish a turn for it against the last
        known board (no need to wait for the next camera frame)."""
        goal = msg.data.strip().upper()
        if goal not in self.GOAL_COLUMN_MAP:
            self.get_logger().warn(f"Ignoring unknown goal '{msg.data}'.")
            return

        self.goal = goal
        self.get_logger().info(f"New mission goal received: {self.goal}")

        # A fresh goal is itself a reason to publish immediately if we can --
        # the downstream consumer needs the turn for the goal it just asked
        # about, even though nothing about the board itself changed. Only
        # do this against a board
        # that was actually detected, never against an empty/stale one.
        if self.board_valid and self.goal in self.current_board:
            self.publish_turn(reason="goal changed")

    # ===================================================================
    # OpenCV board pipeline (algorithm unchanged from detector.py,
    # only moved onto the class so it can run per-frame)
    # ===================================================================

    def process_frame(self, image):
        """Top-level per-frame pipeline: detect board -> warp -> read all
        six columns -> validate -> replace self.current_board -> apply
        change-only publish gate -> publish turn if appropriate.

        Frames are never partially trusted: either all six columns are
        read successfully and current_board is replaced wholesale with
        that fresh reading, or the frame is discarded outright and the
        previous board is left untouched. This is the safer of the two
        options for a robotics competition -- it guarantees current_board
        is always either empty or a real, complete, single-frame reading,
        never a mix of directions read at different times.
        """
        try:
            ordered_pts, _green_mask = self.detect_board(image)
            warped = self.warp_board(image, ordered_pts)
            if DEBUG:
                self._save_debug_image("warped_board", warped)
            columns = self.split_into_columns(warped)
        except RuntimeError:
            # No green board polygon this frame -- board is genuinely gone.
            self.board_present_prev = False
            self._reset_pending_streak()
            self._reset_letter_history()
            return

        board_this_frame = self.read_full_board(columns)
        if not self._is_complete_board(board_this_frame):
            # Board polygon was found but one or more arrows could not be
            # read (glare, motion blur, partial occlusion, ...). Discard
            # this frame rather than acting on incomplete data. We do NOT
            # touch board_present_prev here: a single noisy frame is not
            # the same as the board disappearing, so it must not be able
            # to fake a "disappeared then reappeared" cooldown bypass. It
            # does, however, break any in-progress stability streak.
            self.get_logger().warn(
                f"Partial board ({len(board_this_frame)}/{self.NUM_COLUMNS} "
                "columns read) -- ignoring frame.",
                throttle_duration_sec=self.LOG_THROTTLE_SECONDS,
            )
            self._reset_pending_streak()
            return

        if not self._update_pending_streak(board_this_frame):
            # Complete board, but it hasn't repeated identically for
            # STABILITY_FRAMES_REQUIRED frames yet -- not trustworthy for
            # navigation yet. current_board (if any) is left untouched.
            return

        if DEBUG and self.pending_streak == self.STABILITY_FRAMES_REQUIRED:
            # Print exactly once, at the frame where the streak first
            # crosses the required threshold -- not on every subsequent
            # frame the same board keeps holding. That is what keeps this
            # "one compact line per accepted board" instead of per-frame
            # spam.
            self._debug_print_accepted(board_this_frame)

        self.current_board = board_this_frame  # full replace, never merge
        self.board_valid = True
        self._handle_valid_board()

    def _debug_print_accepted(self, board):
        """DEBUG-only: one compact multi-line print for a newly-accepted
        board, in fixed A/B/C/X/Y/Z order, plus the two tunable constants
        that produced this acceptance. Called at most once per accepted
        board (see the call site in process_frame) -- never per frame."""
        lines = ["Board accepted"]
        for letter in self.GOAL_COLUMN_MAP:  # fixed order: A, B, C, X, Y, Z
            lines.append(f"{letter}={board[letter][0]}")
        lines.append(f"history={self.LETTER_HISTORY_LENGTH}")
        lines.append(f"streak={self.pending_streak}")
        print("\n".join(lines))

    def _reset_pending_streak(self):
        self.pending_signature = None
        self.pending_streak = 0

    def _reset_letter_history(self):
        for history in self.letter_history.values():
            history.clear()

    # ---------------------------------------------------------------
    # FIX 4: debug mode helpers. These are only ever called from sites
    # already guarded by `if DEBUG:`, so when DEBUG is False none of this
    # code runs at all -- zero performance penalty.
    # ---------------------------------------------------------------

    def _debug_dir(self):
        os.makedirs(DEBUG_DIR, exist_ok=True)
        return DEBUG_DIR

    def _save_debug_image(self, name, image):
        path = os.path.join(self._debug_dir(), f"{name}.png")
        cv2.imwrite(path, image)

    def _save_debug_column(self, letter, arrow_roi, debug_info):
        """Saves the arrow ROI and its binary mask (with the classification
        bounding box drawn on it) for a single column, for later tuning."""
        self._save_debug_image(f"column_{letter}_roi", arrow_roi)

        binary = debug_info.get("binary")
        if binary is None:
            return
        mask_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        x, y, w, h = debug_info["bbox"]
        cv2.rectangle(mask_vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
        self._save_debug_image(f"column_{letter}_mask", mask_vis)

    def _update_pending_streak(self, board_this_frame):
        """Requirement 6 (board stability): advances the consecutive-match
        streak for `board_this_frame` and returns True once that exact
        board has now been read STABILITY_FRAMES_REQUIRED frames in a row.
        A board that differs from the in-progress candidate restarts the
        streak at 1 rather than accumulating across different readings."""
        signature = tuple(sorted(board_this_frame.items()))

        if signature == self.pending_signature:
            self.pending_streak += 1
        else:
            self.pending_signature = signature
            self.pending_streak = 1

        return self.pending_streak >= self.STABILITY_FRAMES_REQUIRED

    def read_full_board(self, columns):
        """Runs the arrow detector on all six columns and returns a dict
        of letter -> direction for every column that was read successfully
        this frame. May return fewer than six entries; completeness is
        checked separately by _is_complete_board.

        The raw single-frame classification for each letter is first
        pushed into that letter's own rolling history. The value that
        actually goes into the returned board is _stable_letter_vote()
        over that history -- not the raw single-frame reading, and not a
        guessed/tie-broken value. If a letter's history is not yet full,
        or is full but still genuinely split (still oscillating), it
        contributes nothing to `board` this frame: that letter is simply
        left out, which is what makes the board incomplete and keeps the
        whole thing waiting (see _is_complete_board / process_frame).
        This runs right after arrow detection and before the board dict
        is handed back to process_frame (which is what eventually
        replaces current_board). It adds no extra latency: each column is
        still read exactly once per frame, and the vote is a cheap
        O(history length) count.
        """
        board = {}
        for index in range(self.NUM_COLUMNS):
            letter = self.COLUMN_LETTER_MAP[index]
            column_img, _x_offset = columns[index]
            try:
                arrow_roi, _roi_x, _roi_y = self.extract_arrow_roi(column_img)
                direction, debug_info = self.detect_arrow(arrow_roi)
            except RuntimeError as exc:
                self.get_logger().warn(
                    f"Column '{letter}' unreadable: {exc}",
                    throttle_duration_sec=self.LOG_THROTTLE_SECONDS,
                )
                continue

            if DEBUG:
                self._save_debug_column(letter, arrow_roi, debug_info)

            self.letter_history[letter].append(direction)
            stable_direction = self._stable_letter_vote(self.letter_history[letter])
            if stable_direction is not None:
                board[letter] = stable_direction
            # else: letter not yet stable this frame -- omit it; the board
            # becomes incomplete and process_frame will wait rather than
            # guess.
        return board

    def _stable_letter_vote(self, history):
        """Strict-majority vote over a letter's rolling history of raw
        per-frame classifications. Returns the winning direction ONLY if
        that letter is genuinely stable; otherwise returns None and the
        letter is treated as unread this frame.

        "Stable" means both of the following, using only
        LETTER_HISTORY_LENGTH (no other tuning constant):
            1. The history has filled to LETTER_HISTORY_LENGTH -- a
               partially-filled history (e.g. right after the board
               reappears) cannot yet claim stability.
            2. One direction holds a STRICT majority (count > half the
               history length) -- not merely the most frequent value.

        There is deliberately no tie-break by recency here any more: a
        letter that is still oscillating (e.g. RIGHT, LEFT, RIGHT,
        STRAIGHT with no direction holding more than half the window)
        must not be resolved by guessing which one is "freshest". Per the
        requirement, an unstable letter simply keeps waiting -- it is
        left out of this frame's board entirely, which makes the board
        incomplete and therefore un-publishable (see _is_complete_board /
        process_frame) until the oscillation naturally settles.
        """
        if len(history) < self.LETTER_HISTORY_LENGTH:
            return None

        counts = Counter(history)
        direction, count = counts.most_common(1)[0]
        if count > len(history) / 2:
            return direction
        return None

    def _is_complete_board(self, board):
        """A board is only trustworthy for navigation if every one of the
        six columns was read this frame."""
        return len(board) == self.NUM_COLUMNS

    # ---------------------------------------------------------------
    # Board detection
    # ---------------------------------------------------------------

    def order_points(self, pts):
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

    def detect_board(self, image):
        """
        Locate the large green board in `image` using HSV thresholding,
        morphological cleanup, and largest-contour + polygon approximation.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.GREEN_HSV_LOWER, self.GREEN_HSV_UPPER)

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
            rect = cv2.minAreaRect(largest)
            quad = cv2.boxPoints(rect)

        ordered_pts = self.order_points(np.array(quad, dtype="float32"))

        return ordered_pts, mask

    # ---------------------------------------------------------------
    # Perspective warp
    # ---------------------------------------------------------------

    def warp_board(self, image, ordered_pts, width=None, height=None):
        """Perspective-warp the quadrilateral defined by `ordered_pts` into a
        flat width x height rectangle."""
        width = self.WARPED_WIDTH if width is None else width
        height = self.WARPED_HEIGHT if height is None else height

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

    # ---------------------------------------------------------------
    # Column splitting
    # ---------------------------------------------------------------

    def split_into_columns(self, warped, num_columns=None):
        """Split the warped board into `num_columns` equal-width vertical
        slices. Returns a list of (column_image, x_offset) tuples."""
        num_columns = self.NUM_COLUMNS if num_columns is None else num_columns
        height, width = warped.shape[:2]
        col_width = width // num_columns

        columns = []
        for i in range(num_columns):
            x_start = i * col_width
            x_end = width if i == num_columns - 1 else (i + 1) * col_width
            column_img = warped[0:height, x_start:x_end]
            columns.append((column_img, x_start))

        return columns

    # ---------------------------------------------------------------
    # Arrow ROI extraction
    # ---------------------------------------------------------------

    def extract_arrow_roi(self, column_img, letter_fraction=None):
        """Crop the arrow region away from the panel borders."""
        letter_fraction = (
            self.LETTER_REGION_FRACTION if letter_fraction is None else letter_fraction
        )
        height, width = column_img.shape[:2]

        top = int(height * 0.40)
        bottom = int(height * 0.95)
        left = int(width * 0.20)
        right = int(width * 0.80)

        arrow_roi = column_img[top:bottom, left:right]
        return arrow_roi, left, top

    # ---------------------------------------------------------------
    # Arrow direction detection
    # ---------------------------------------------------------------

    def _binarize_arrow(self, arrow_roi):
        """Convert the arrow ROI into a clean binary mask."""
        gray = cv2.cvtColor(arrow_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        return thresh

    def _largest_arrow_contour(self, binary):
        """
        Return the arrow contour in the binary mask, filtering out thin
        full-height border/divider slivers first.
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

    def _classify_arrow_by_mass(self, binary, contour, edge_fraction=0.4, center_gap=0.2):
        """
        Classify arrow direction using pixel-mass distribution instead of
        contour-tip geometry.
        """
        if binary.size == 0 or cv2.countNonZero(binary) == 0:
            raise RuntimeError("Empty arrow mask; nothing to classify.")

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

        col_sum = (binary > 0).sum(axis=0).astype(np.float64)
        row_sum = (binary > 0).sum(axis=1).astype(np.float64)

        horizontal_spread = np.count_nonzero(col_sum)
        vertical_spread = np.count_nonzero(row_sum)

        edge_px = max(1, int(round(w * edge_fraction)))
        gap_px = int(round(w * center_gap / 2))
        mid = w // 2

        left_band = col_sum[:edge_px]
        right_band = col_sum[max(0, w - edge_px):]
        if edge_px > mid - gap_px:
            left_band = col_sum[: max(1, mid - gap_px)]
            right_band = col_sum[min(w - 1, mid + gap_px):]

        left_mass = float(left_band.sum())
        right_mass = float(right_band.sum())

        # FIX 3: the Y column's arrow was being misread as STRAIGHT.
        # Investigation: `is_tall_narrow` was deciding STRAIGHT from the
        # bounding-box aspect ratio (h > 1.4 * w) alone. The Y panel's
        # arrow contour bounding box happens to sit just above that 1.4
        # ratio -- but its pixel mass is NOT concentrated on the vertical
        # centerline the way a real STRAIGHT arrow's is; it is skewed
        # left/right like every other directional arrow. A bare bbox
        # ratio can't tell a "genuinely tall shaft" apart from a
        # "left/right arrow that happens to occupy a tall-ish crop", so it
        # was tripping on Y even though the pixel-mass evidence (the same
        # left_mass/right_mass split already computed above) clearly
        # favoured a direction. The fix keeps the same aspect-ratio
        # heuristic but only lets it decide STRAIGHT when the mass split
        # is itself close to balanced (i.e. no clear left/right skew);
        # whenever the mass distribution is decisively lopsided, that mass
        # evidence -- not the raw bbox ratio -- determines the direction.
        # This changes no thresholds for A/B/C/X/Z, whose arrows already
        # have a strongly imbalanced mass split and so are unaffected.
        is_tall_narrow = h > 1.4 * w
        mass_total = left_mass + right_mass
        mass_imbalance = (
            abs(left_mass - right_mass) / mass_total if mass_total > 0 else 0.0
        )
        MASS_IMBALANCE_THRESHOLD = 0.15  # below this, treat mass as "balanced"

        if is_tall_narrow and mass_imbalance < MASS_IMBALANCE_THRESHOLD:
            direction = "STRAIGHT"
        else:
            direction = "LEFT" if left_mass > right_mass else "RIGHT"

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

    def detect_arrow(self, arrow_roi):
        """Detect arrow direction from the selected ROI using pixel-mass
        distribution (robust to blur / anti-aliasing / minor contour noise)."""
        binary = self._binarize_arrow(arrow_roi)
        contour = self._largest_arrow_contour(binary)
        direction, debug_info = self._classify_arrow_by_mass(binary, contour)
        debug_info["binary"] = binary
        debug_info["contour"] = contour
        return direction, debug_info

    def navigation_decision(self, direction):
        """Map a detected arrow direction to a navigation command string."""
        if direction not in self.NAVIGATION_TABLE:
            raise ValueError(f"Unknown arrow direction: {direction}")
        return self.NAVIGATION_TABLE[direction]

    # ===================================================================
    # Mission logic: goal lookup, change-only publish gate, publishing
    # ===================================================================

    def _board_signature(self):
        """Signature of the current board. Only ever called while
        self.board_valid is True, so it always represents a real,
        complete detection -- never a stale or partial one."""
        return tuple(sorted(self.current_board.items()))

    def _handle_valid_board(self):
        """FIX 1: called once per frame after a COMPLETE, valid board has
        replaced self.current_board. Publishes ONLY when the mission goal
        changed (handled separately in goal_callback), the detected board
        changed, or the board disappeared and then reappeared. There is no
        cooldown-based republishing any more: as long as nothing changes,
        the previously published turn simply remains valid and the robot
        stays silent -- it does not keep re-publishing the same turn."""
        signature = self._board_signature()

        board_reappeared = not self.board_present_prev
        board_changed = signature != self.last_board_signature

        self.board_present_prev = True
        self.last_board_signature = signature

        if board_reappeared or board_changed:
            reason = "board reappeared" if board_reappeared else "board changed"
            self.publish_turn(reason=reason)

    def publish_turn(self, reason="update"):
        """Publishes the turn direction for the current goal, if the goal
        is known and a valid (complete) board is available and contains
        that goal. This is the single place that writes to /mission/turn."""
        if not self.board_valid or self.goal is None or self.goal not in self.current_board:
            return

        direction = self.current_board[self.goal]
        msg = String()
        msg.data = direction
        self.turn_pub.publish(msg)

        self.get_logger().info(f"Published turn '{direction}' for goal '{self.goal}' ({reason}).")

    # ===================================================================
    # RESERVED FOR FUTURE MISSION LOGIC
    # ===================================================================
    # Add later stages here as additional methods on DetectNode, each with
    # its own topic(s)/subscription(s) set up in __init__, following the
    # same pattern used above (small method, own state on self, no
    # globals). None of this touches the OpenCV pipeline or the board /
    # cooldown logic already implemented.
    #
    #   - QR detection
    #   - Municipality server communication
    #   - Hospital verification
    #   - Parking
    #   - Mission state machine (sequencing goals, tracking mission phase)
    # ===================================================================


def main(args=None):
    rclpy.init(args=args)
    node = DetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
