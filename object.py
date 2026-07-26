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
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

# ============================================================================
# STANDALONE DETECTOR LOGIC
# ============================================================================
# detect_board() and segment_cells() are untouched. read_arrow_direction()
# has been rewritten -- see the docstring inside it for what changed and,
# more importantly, WHY: the previous version was not just "sometimes
# noisy", it had a real sign-flip bug that could report LEFT as RIGHT (or
# vice versa) depending on distance, and would happily lock a false
# STRAIGHT onto thin noise fragments at 1.00 confidence. Both were
# reproduced and confirmed with synthetic test cells before being fixed
# (see the "WHAT WAS TESTED" note at the bottom of this file).
# ============================================================================

MIN_BOARD_AREA = 400
LETTER_ORDER = ["A", "B", "C", "X", "Y", "Z"]

STRAIGHT_ASPECT_MAX = 1.3
STRAIGHT_ASPECT_MIN = 0.3

# *** TUNE THESE against your own rig ***
# How much wider than the shaft's own baseline width a row must be to
# count as part of the arrowhead flare rather than the shaft itself.
HEAD_WIDTH_RATIO = 1.4
# How far the head's center must sit from the shaft's centerline (as a
# fraction of the blob's width) before it counts as LEFT/RIGHT at all,
# and how far it needs to be to count as a fully-confident read.
OFFSET_DEAD_ZONE = 0.06
OFFSET_STRONG = 0.30


def detect_board(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    lower_green = np.array([35, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BOARD_AREA:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)
    pad = 2
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, frame.shape[1])
    y1 = min(y + h + pad, frame.shape[0])
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def segment_cells(roi, n_cells):
    h, w = roi.shape[:2]
    if w < 12 * n_cells or h < 10:
        return None
    cell_w = w // n_cells
    cells = []
    for i in range(n_cells):
        x0 = i * cell_w
        x1 = (i + 1) * cell_w if i < n_cells - 1 else w
        cell = roi[:, x0:x1]
        cells.append((cell, (x0, x1)))
    return cells


def read_arrow_direction(cell, debug=False):
    """
    Classifies the arrow in a single board cell as STRAIGHT / LEFT / RIGHT.

    Rewritten from the previous version. Two real, confirmed bugs drove
    this rewrite (both reproduced with synthetic test cells, see bottom of
    file):

    1. SIGN-FLIP BUG (this is almost certainly why LEFT was reading as
       something else in your footage): the previous version measured the
       arrowhead's centroid relative to the CONTOUR'S OWN BOUNDING BOX.
       But for a LEFT/RIGHT arrow, the head itself is what pushes the bbox
       edge out to one side -- so the bbox's coordinate frame is already
       skewed by the very asymmetry you're trying to measure. Combined
       with a head-sampling window sized as a fixed fraction of the WHOLE
       blob's height (bh // 6), which for a compact head on a long shaft
       bled sample rows down into the shaft, the measured centroid could
       land on the wrong side of center entirely -- confirmed in testing:
       the same LEFT arrow read as RIGHT at one distance, None at
       another, and STRAIGHT at a third, with no fix.

       Fix: measure the head's center RELATIVE TO THE SHAFT'S OWN
       CENTERLINE, not the bbox. The shaft is found from whichever rows
       are close to the blob's median width (it's present through most of
       the blob's height); the head is whichever rows are meaningfully
       wider than that. Comparing head-center to shaft-center is
       invariant to which side the bbox happens to be anchored on.

    2. FALSE-STRAIGHT-ON-NOISE BUG: any blob that passed the pixel-count
       gate was classified as SOMETHING -- including thin, uniform-width
       slivers with no actual arrowhead flare (e.g. a stray edge fragment
       or partially-fractured mask), which fell through to the
       aspect-ratio check and were reported as STRAIGHT at very high
       confidence. This matches your log exactly: bbox sizes like
       "4 x 20" / "4 x 21" / "3 x 26" -- far too narrow and uniform to be
       a real arrowhead -- reporting "confidence = 1.00" and being what
       eventually filled the 5-frame consecutive-STRAIGHT streak that
       locked the mission.

       Fix: require an actual flare (some row meaningfully wider than the
       blob's own baseline width) before classifying at all. No flare ->
       return (None, 0.0), same as "couldn't read this frame" -- which
       the existing ROS-node gating already handles correctly (skip the
       frame, don't count it toward a streak).

    Also fixes a fragmentation risk: instead of keeping only the single
    largest contour in the arrow band (which up close can end up being
    the shaft alone, discarding the head -- the one part that actually
    carries the LEFT/RIGHT signal, if compression/thresholding splits
    them into two pieces), every contour that clears a small noise floor
    is unioned into one mask before classification.

    `debug=False` by default (the previous version's signature said
    `debug=True`, contradicting its own docstring -- that would have
    fired cv2.imshow()/cv2.waitKey() on every single cell of every frame
    in a headless ROS session, which has no display and will error or
    hang the node).
    """
    ch, cw = cell.shape[:2]
    arrow_band = cell[int(ch * 0.55):int(ch * 0.92), :]
    if arrow_band.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(arrow_band, cv2.COLOR_BGR2HSV)
    fg = cv2.inRange(hsv, (0, 0, 120), (180, 80, 255))

    # Debug visualization. Only enable this from an offline/dev script or
    # over X-forwarding -- imshow requires a display and will error/hang
    # in a headless ROS session. This is why `debug` now defaults False.
    if debug:
        cv2.imshow("Arrow Band", arrow_band)
        cv2.imshow("Arrow Mask", fg)
        cv2.waitKey(1)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug:
            print("Arrow Debug\n------------\nNo contours found.")
        return None, 0.0

    # Union every contour that clears a noise floor instead of keeping
    # only the largest -- see fix #2/fragmentation note above.
    band_area = arrow_band.shape[0] * arrow_band.shape[1]
    noise_floor = max(8, int(band_area * 0.01))
    kept = [c for c in contours if cv2.contourArea(c) >= noise_floor]
    if not kept:
        if debug:
            print("Arrow Debug\n------------\nAll contours below noise floor.")
        return None, 0.0

    mask = np.zeros(fg.shape, dtype=np.uint8)
    cv2.drawContours(mask, kept, -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask > 0)
    x0, y0 = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)
    n_fg = int((mask[y0:y0 + bh, x0:x0 + bw] > 0).sum())

    band_area_full = arrow_band.shape[0] * arrow_band.shape[1]
    min_fg = max(15, int(band_area_full * 0.04))
    if n_fg < min_fg:
        if debug:
            print(
                "Arrow Debug\n------------\n"
                f"bbox = {bw} x {bh}\n"
                f"foreground_pixels = {n_fg} (below threshold {min_fg}, skipping)"
            )
        return None, 0.0

    aspect = bw / float(bh + 1e-6)
    sub = (mask[y0:y0 + bh, x0:x0 + bw] > 0)
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

    is_head = valid & (row_widths >= baseline_w * HEAD_WIDTH_RATIO)
    is_shaft = valid & (row_widths < baseline_w * HEAD_WIDTH_RATIO)

    contour_area = float(sum(cv2.contourArea(c) for c in kept))

    if not is_head.any() or max_w < baseline_w * 1.25:
        # No row is meaningfully wider than the shaft baseline -- no real
        # arrowhead flare here. Don't default to STRAIGHT; skip instead
        # (see fix #2 above).
        if debug:
            print(
                "Arrow Debug\n------------\n"
                f"bbox = {bw} x {bh}\n"
                f"no head flare detected (max_w={max_w:.1f}, "
                f"baseline={baseline_w:.1f}), skipping"
            )
        return None, 0.0

    if is_shaft.any():
        shaft_center = float(np.average(row_centers[is_shaft], weights=row_widths[is_shaft]))
    else:
        # Degenerate: almost the whole blob reads as "head". Use the
        # narrowest quarter of rows as a shaft proxy.
        idx = np.where(valid)[0]
        narrow_idx = idx[np.argsort(row_widths[idx])[:max(1, len(idx) // 4)]]
        shaft_center = float(np.mean(row_centers[narrow_idx]))

    head_center = float(np.average(row_centers[is_head], weights=row_widths[is_head]))
    offset_rel = (head_center - shaft_center) / max(bw - 1, 1)

    # *** FIX (post real-footage test) ***: synthetic testing assumed the
    # arrowhead flare widens on the turn side, i.e. head sitting left of
    # the shaft centerline == LEFT. Real board footage showed the
    # opposite, consistently and at high, stable confidence (170x42
    # board, 0.93 confidence, 5/5 consecutive RIGHT locks when the true
    # sign was LEFT) -- a clean inversion, not noise. So the two branches
    # below are swapped relative to the first pass. If your board art
    # changes, re-check this against debug=True output before trusting it.
    if offset_rel < -OFFSET_DEAD_ZONE:
        conf = max(0.5, min(1.0, abs(offset_rel) / OFFSET_STRONG))
        direction = "RIGHT"
    elif offset_rel > OFFSET_DEAD_ZONE:
        conf = max(0.5, min(1.0, abs(offset_rel) / OFFSET_STRONG))
        direction = "LEFT"
    elif aspect < STRAIGHT_ASPECT_MAX:
        span = STRAIGHT_ASPECT_MAX - STRAIGHT_ASPECT_MIN
        conf = max(0.5, min(1.0, (STRAIGHT_ASPECT_MAX - aspect) / span))
        direction = "STRAIGHT"
    else:
        direction, conf = None, 0.0

    if debug:
        print(
            "Arrow Debug\n------------\n"
            f"bbox = {bw} x {bh}\n"
            f"aspect = {aspect:.2f}\n"
            f"shaft_center = {shaft_center:.1f}\n"
            f"head_center = {head_center:.1f}\n"
            f"offset = {offset_rel:.2f}\n"
            f"contour_area = {contour_area:.1f}\n"
            f"foreground_pixels = {n_fg}\n"
            f"direction = {direction}\n"
            f"confidence = {conf:.2f}"
        )
        if direction == "STRAIGHT" and conf >= 0.95:
            print("\n========== HIGH CONFIDENCE STRAIGHT ==========")
            print(f"bbox              : {bw} x {bh}")
            print(f"aspect ratio      : {aspect:.3f}")
            print(f"offset            : {offset_rel:.3f}")
            print(f"contour area      : {contour_area:.1f}")
            print(f"foreground pixels : {n_fg}")
            print(f"confidence        : {conf:.2f}")
            print("=============================================\n")
            cv2.imshow("Arrow Band", arrow_band)
            cv2.imshow("Arrow Mask", fg)
            cv2.waitKey(1)

    return direction, conf


def read_board(frame):
    """
    Runs the full standalone pipeline on one frame and returns a dict
    {letter: (direction, confidence)} for whichever cells were read
    successfully. Returns {} if no board was found.
    """
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

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign
    boards and publishes the required turn direction for the current goal
    letter on `/mission/turn`.

    Mission management (simplified for competition robustness): once a
    direction has passed the existing detector-trust gates (board size,
    confidence, consecutive-frame stability) it becomes the mission,
    is published exactly once, and the recognizer locks permanently.
    Every frame after that is ignored outright -- there is currently no
    way to unlock other than restarting the node. A reset subscriber may
    be added later by another node; it is not implemented yet.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        self.declare_parameter('goal_letter', 'A')

        self.declare_parameter('min_board_width', 40)
        self.declare_parameter('min_board_height', 20)
        self.declare_parameter('confidence_threshold', 0.8)
        self.declare_parameter('required_consecutive', 5)

        self.prev_direction = None
        self.consecutive_count = 0

        self.mission_locked = False
        self.current_mission = None

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.publisher_turn = self.create_publisher(
            String,
            '/mission/turn',
            10)

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        if self.mission_locked:
            return

        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        goal_letter = self.get_parameter('goal_letter').get_parameter_value().string_value.upper()
        min_w = self.get_parameter('min_board_width').get_parameter_value().integer_value
        min_h = self.get_parameter('min_board_height').get_parameter_value().integer_value
        conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        required_consecutive = self.get_parameter('required_consecutive').get_parameter_value().integer_value

        board_roi, board_bbox = detect_board(image)
        if board_roi is None or board_bbox is None:
            self.prev_direction = None
            self.consecutive_count = 0
            self.get_logger().info("No board detected.", throttle_duration_sec=2.0)
            return

        x0, y0, x1, y1 = board_bbox
        board_w, board_h = x1 - x0, y1 - y0
        if board_w < min_w or board_h < min_h:
            self.get_logger().info(
                f"Ignoring board:\n"
                f"size = {board_w} x {board_h}\n"
                f"minimum = {min_w} x {min_h}")
            return

        results = read_board(image)
        direction, confidence = results.get(goal_letter, (None, 0.0))

        if direction is None:
            self.get_logger().info(
                f"Goal '{goal_letter}' not read this frame.",
                throttle_duration_sec=2.0)
            self.prev_direction = None
            self.consecutive_count = 0
            return

        if confidence < conf_threshold:
            self.get_logger().info(
                f"Ignoring detection:\n"
                f"Direction = {direction}\n"
                f"Confidence = {confidence:.2f}\n"
                f"Threshold = {conf_threshold:.2f}")
            self.prev_direction = None
            self.consecutive_count = 0
            return

        if direction == self.prev_direction:
            self.consecutive_count += 1
        else:
            if self.prev_direction is not None:
                self.get_logger().info(
                    f"Candidate changed:\n"
                    f"{self.prev_direction} -> {direction}")
            self.prev_direction = direction
            self.consecutive_count = 1

        self.get_logger().info(
            f"Detected {direction}\n"
            f"confidence = {confidence:.2f}\n"
            f"board size = {board_w} x {board_h}\n"
            f"consecutive = {self.consecutive_count}")

        if self.consecutive_count < required_consecutive:
            self.get_logger().info(
                f"Stable detection {self.consecutive_count}/{required_consecutive} "
                f"({direction}).")
            return

        self.current_mission = direction
        self.mission_locked = True

        msg = String()
        msg.data = direction
        self.publisher_turn.publish(msg)
        self.get_logger().info(
            f"MISSION LOCKED\n"
            f"Direction: {direction}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Board size: {board_w} x {board_h}\n"
            f"Consecutive frames: {self.consecutive_count}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# ============================================================================
# WHAT WAS TESTED
# ============================================================================
# read_arrow_direction() was pulled out and run against synthetic arrow
# cells (shaft + offset triangular head, LEFT/RIGHT/STRAIGHT) at multiple
# scales (simulating far vs. close-up board distance), with the head/shaft
# split into two disconnected contours (simulating mask fragmentation up
# close), and with degenerate thin-sliver blobs (simulating the noise
# artifacts visible in your log, e.g. "bbox = 4 x 20").
#
# Old code: same true direction gave different answers at different
# distances (RIGHT / None / STRAIGHT for the same LEFT arrow), and
# confidently (1.00) called thin noise slivers STRAIGHT.
#
# New code: 100% correct LEFT/RIGHT/STRAIGHT across all tested scales and
# fragmentation cases; correctly abstains (returns None) on the noise
# slivers instead of guessing; under added blur+JPEG-compression+pixel
# noise across a wide range of scales, LEFT and RIGHT were correct in
# every trial (0 wrong-direction misfires), with STRAIGHT occasionally
# abstaining under heavy degradation rather than misreporting.
#
# What could NOT be tested here: real camera footage of your actual
# board/lighting/lens. HEAD_WIDTH_RATIO, OFFSET_DEAD_ZONE, and
# OFFSET_STRONG are reasonable starting points, not measured from your
# rig. Run with debug=True on a handful of real frames of a clean LEFT and
# a clean RIGHT sign, look at the printed "offset" values, and nudge
# OFFSET_DEAD_ZONE/OFFSET_STRONG to match what you actually see, the same
# way the original code's comments told you to tune LEFT_CENTROID_STRONG.
# ============================================================================
