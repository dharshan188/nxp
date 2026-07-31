import math
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy, LaserScan
from synapse_msgs.msg import EdgeVectors
from std_msgs.msg import String

QOS_PROFILE_DEFAULT = 10

SPEED_MIN = 0.0
SPEED_MAX = 1.0

TURN_MIN = -1.0
TURN_MAX = 1.0


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # ---------------- Parameters (all live-tunable via ros2 param set) --------
        self.declare_parameter('steer_sign', -1.0)
        self.declare_parameter('Kp', 0.55)
        self.declare_parameter('Ki', 0.0)
        self.declare_parameter('Kd', 0.18)
        self.declare_parameter('lookahead_blend', 0.6)
        self.declare_parameter('lane_width_px', 240.0)
        self.declare_parameter('learn_lane_width', True)
        self.declare_parameter('single_vector_side_margin', 0.20)
        self.declare_parameter('speed_straight', 0.45)
        self.declare_parameter('speed_sharp', 0.22)
        self.declare_parameter('speed_lost', 0.15)
        self.declare_parameter('steer_alpha', 0.55)
        self.declare_parameter('speed_alpha', 0.15)
        self.declare_parameter('no_vector_hold', 0.6)
        self.declare_parameter('debug_log', True)
        self.declare_parameter('obstacle_enable', True)
        self.declare_parameter('obstacle_trigger_dist', 0.90)
        self.declare_parameter('obstacle_clear_dist', 1.30)
        self.declare_parameter('obstacle_fov_deg', 60.0)
        self.declare_parameter('obstacle_turn_gain', 0.9)
        self.declare_parameter('obstacle_speed', 0.20)
        # --- edge-safety margin so the aim point never sits right on the curb ---
        self.declare_parameter('turn_edge_margin_px', 35.0)
        # --- junction lane-width ratio that triggers "wide crossing" handling ---
        self.declare_parameter('junction_width_ratio', 1.8)

        # NEW: Straight-intersection state-machine parameters --------------------------------
        # Width ratio to ENTER intersection mode (wide opening)
        self.declare_parameter('intersect_entry_width_ratio', 1.6)
        # Width ratio to EXIT intersection mode (must be back to normal lane)
        self.declare_parameter('intersect_exit_width_ratio', 1.25)
        # Fractional spike in measured lane width that counts as "sudden open"
        self.declare_parameter('intersect_width_spike_ratio', 1.35)
        # Number of consecutive "good" frames required before leaving intersection mode
        self.declare_parameter('intersect_stable_frames', 6)
        # Max time (s) we are allowed to stay in intersection dead-reckoning (safety timeout)
        self.declare_parameter('intersect_max_time', 4.0)
        # How long (s) to be without vectors before intersection mode engages (short loss at entry)
        self.declare_parameter('intersect_no_vec_time', 0.18)
        # Proportional gain on stored heading while in intersection (turn to follow stored road angle)
        self.declare_parameter('intersect_heading_gain', 0.30)
        # Gain on cross-track error (lane_center - image_center) while in intersection (keeps us centered)
        self.declare_parameter('intersect_cte_gain', 0.20)
        # How strongly to blend new two-vector heading toward stored heading while still inside (0 = trust new 100% is wrong, use for gentle correction only)
        self.declare_parameter('intersect_heading_blend', 0.15)
        # Forward speed while traversing the intersection
        self.declare_parameter('intersect_speed', 0.38)
        # Minimum number of good (non-intersection) two-vector samples before we trust width-spike detection (avoids startup false trigger)
        self.declare_parameter('intersect_width_samples_for_spike', 6)
        # NEW: minimum elapsed time since last two-vector sighting before single-vector
        # triggers intersection entry. Prevents false triggering on trivial occlusions
        # (gate posts, signs) where one edge briefly drops out for one frame.
        self.declare_parameter('intersect_one_vec_time', 0.25)
        # NEW: cooldown after a timeout/watchdog exit during which we refuse to re-enter
        # via no_vec/one_vec (wide/spike still allowed because they require two vectors
        # and thus fresh trustworthy data). Stops the re-entry loop seen in the log.
        self.declare_parameter('intersect_cooldown_after_timeout', 1.5)
        # NEW: max allowed heading magnitude (deg) at entry. Headings steeper than this
        # are considered garbage (e.g. cross-street edge, post, near-vertical line) and
        # we fall back to 0 = straight, rather than slamming the wheel.
        self.declare_parameter('intersect_max_entry_heading_deg', 30.0)
        # NEW: heading-jump threshold (deg) for single-vector intersection entry.
        # If a single edge's heading differs from our last known road heading by more
        # than this, the detector has latched onto the cross-street curb -> enter INT.
        self.declare_parameter('intersect_heading_jump_deg', 20.0)
        # NEW: how old the last two-vector heading memory is allowed to be before we
        # refuse single-vector entry (if we haven't seen a proper lane in this long,
        # we don't have a heading worth locking to).
        self.declare_parameter('intersect_memory_max_age', 1.0)
        # NEW: hard clamp (deg) on the locked heading when entering intersection.
        # STRAIGHT mission = go nearly straight through; anything steeper than this
        # at entry is curve/cross-street bias, not a real turn command.
        self.declare_parameter('intersect_lock_max_heading_deg', 8.0)
        # NEW: slow-EMA alpha for the "long-term road heading" that we lock to.
        # Small = smoother/more resistant to curve bias at the mouth of intersection.
        self.declare_parameter('intersect_slow_ema_alpha', 0.07)
        # NEW: fast-EMA alpha for jump detection (needs to track current edge quickly
        # so we see the sudden cross-street latch).
        self.declare_parameter('intersect_fast_ema_alpha', 0.4)

        self._reload_params()

        # ---------------- Runtime state ----------------
        self.error = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        self.vectors_available = False
        self.last_vector_time = None
        self.last_single_side = 'LEFT'
        self.learned_lane_width = self.lane_width_px
        self.target_turn = 0.0
        self.target_speed = 0.0
        self.filtered_turn = 0.0
        self.filtered_speed = 0.0
        self.last_good_turn = 0.0
        self.obstacle_detected = False
        self.obstacle_turn = 0.0
        self.nearest_dist = float('inf')
        self._tick = 0
        self.current_mission = "STRAIGHT"
        self._straight_junction_side = None

        # NEW: Straight-intersection state machine runtime state ------------------------------
        # Are we currently inside the STRAIGHT_INTERSECTION state?
        self._in_intersection = False
        # Monotonic timestamp (time.time()) when we first entered intersection mode
        self._intersection_entry_time = None
        # Stored road heading (radians, image-space) captured at entry. 0 = straight ahead in image;
        # positive = road curves/angles to the right of straight-down; negative = left.
        # Computed from raw edge-vector geometry (NOT PID output, NOT steering command).
        self._intersection_heading = 0.0
        # Stored cross-track error normalized [-1, 1] at the moment of entry: (lane_center - img_center)/img_center
        self._intersection_cte = 0.0
        # Counter of consecutive frames where two vectors AND normal width are seen. When it hits
        # intersect_stable_frames we exit intersection mode.
        self._intersection_stable_count = 0
        # EMA-smoothed measured lane width used for spike detection (separate from learned lane width).
        self._width_ema = self.lane_width_px
        # Number of valid two-vector samples used to populate _width_ema so far. We only fire
        # spike detection once we have enough samples to avoid a startup false positive.
        self._width_ema_samples = 0
        # Last heading (radians) computed from a "trustworthy" two-vector measurement in NORMAL mode.
        # Used as a fallback if we enter intersection on a no-vector frame.
        self._last_good_heading = 0.0
        # Last normalized cte from a trustworthy NORMAL-mode frame; fallback for no-vector entry.
        self._last_good_cte = 0.0
        # Running FAST EMA of edge heading in NORMAL mode (quickly tracks current edge,
        # used to DETECT the sudden jump onto a cross-street curb).
        self._heading_ema_fast = 0.0
        # Running SLOW EMA of edge heading in NORMAL mode (smoothed over ~1 s, represents
        # the long-range road direction / "straight ahead" reference we LOCK TO when
        # entering intersection, so approach-curve bias doesn't make us turn hard).
        self._heading_ema_slow = 0.0
        # Both EMAs populated yet?
        self._heading_ema_init = False
        # Timestamp (time.time()) of the last time we saw TWO vectors in NORMAL mode.
        # Used to distinguish "just entered intersection, lost one edge" from
        # "one edge was always occluded (post, sign)" — both give count==1.
        self._last_two_vec_time = None
        # Timestamp until which weak triggers (one_vec / no_vec) are REFUSED after
        # a timeout/watchdog exit. Prevents the infinite re-entry loop when heading
        # memory is stale/bad. Wide/spike triggers (which require fresh two-vector data)
        # are still allowed.
        self._intersect_cooldown_until = 0.0

        # ---------------- ROS plumbing ----------------
        self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/mission/turn', self.mission_callback, QOS_PROFILE_DEFAULT)

        self.pub_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

        self.create_timer(0.033, self.control_loop)
        self.create_timer(1.0, self._reload_params)

        self.get_logger().info("Lane-following controller with STRAIGHT_INTERSECTION state loaded.")

    # ------------------------------------------------------------------
    # Parameter reload (called once on init and every 1s thereafter)
    # ------------------------------------------------------------------
    def _reload_params(self):
        g = lambda n: self.get_parameter(n).value

        self.steer_sign = float(g('steer_sign'))
        self.Kp = g('Kp')
        self.Ki = g('Ki')
        self.Kd = g('Kd')
        self.lookahead_blend = g('lookahead_blend')
        self.lane_width_px = g('lane_width_px')
        self.learn_lane_width = g('learn_lane_width')
        self.side_margin = g('single_vector_side_margin')
        self.speed_straight = g('speed_straight')
        self.speed_sharp = g('speed_sharp')
        self.speed_lost = g('speed_lost')
        self.steer_alpha = g('steer_alpha')
        self.speed_alpha = g('speed_alpha')
        self.no_vector_hold = g('no_vector_hold')
        self.debug_log = g('debug_log')
        self.obstacle_enable = g('obstacle_enable')
        self.obstacle_trigger_dist = g('obstacle_trigger_dist')
        self.obstacle_clear_dist = g('obstacle_clear_dist')
        self.obstacle_fov_deg = g('obstacle_fov_deg')
        self.obstacle_turn_gain = g('obstacle_turn_gain')
        self.obstacle_speed = g('obstacle_speed')
        self.turn_edge_margin_px = g('turn_edge_margin_px')
        self.junction_width_ratio = g('junction_width_ratio')

        # NEW: load intersection parameters
        self.intersect_entry_width_ratio = g('intersect_entry_width_ratio')
        self.intersect_exit_width_ratio = g('intersect_exit_width_ratio')
        self.intersect_width_spike_ratio = g('intersect_width_spike_ratio')
        self.intersect_stable_frames = int(g('intersect_stable_frames'))
        self.intersect_max_time = g('intersect_max_time')
        self.intersect_no_vec_time = g('intersect_no_vec_time')
        self.intersect_heading_gain = g('intersect_heading_gain')
        self.intersect_cte_gain = g('intersect_cte_gain')
        self.intersect_heading_blend = g('intersect_heading_blend')
        self.intersect_speed = g('intersect_speed')
        self.intersect_width_samples_for_spike = int(g('intersect_width_samples_for_spike'))
        self.intersect_one_vec_time = g('intersect_one_vec_time')
        self.intersect_cooldown_after_timeout = g('intersect_cooldown_after_timeout')
        self.intersect_max_entry_heading_deg = g('intersect_max_entry_heading_deg')
        self.intersect_heading_jump_deg = g('intersect_heading_jump_deg')
        self.intersect_memory_max_age = g('intersect_memory_max_age')
        self.intersect_lock_max_heading_deg = g('intersect_lock_max_heading_deg')
        self.intersect_slow_ema_alpha = g('intersect_slow_ema_alpha')
        self.intersect_fast_ema_alpha = g('intersect_fast_ema_alpha')

    # ------------------------------------------------------------------
    # Mission topic callback
    # ------------------------------------------------------------------
    def mission_callback(self, msg):
        mission = msg.data
        if mission in ["LEFT", "RIGHT", "STRAIGHT"]:
            if mission != self.current_mission:
                self.get_logger().info(f"Mission changed to {mission}")
                self.current_mission = mission
                self._straight_junction_side = None
                # NEW: mission change resets intersection state (we never want to stay locked
                # into a dead-reckoned heading if the operator changes plan).
                self._reset_intersection_state()
                # Also clear heading-EMA memory so we re-prime from the new mission's first frames.
                self._heading_ema_init = False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _aim_x(self, vector):
        p0, p1 = vector[0], vector[1]
        near, far = (p1, p0) if p1.y >= p0.y else (p0, p1)
        b = self.lookahead_blend
        return (1.0 - b) * near.x + b * far.x

    @staticmethod
    def _mean_x(vector):
        return (vector[0].x + vector[1].x) / 2.0

    def _clamped_offset(self, offset, lane_width):
        """Keep the aim point at least turn_edge_margin_px away from either edge."""
        margin = self.turn_edge_margin_px
        if lane_width <= 2.0 * margin:
            return lane_width / 2.0
        return max(margin, min(lane_width - margin, offset))

    # NEW: -----------------------------------------------------------------
    # Intersection state-machine helpers
    # ------------------------------------------------------------------
    def _reset_intersection_state(self):
        """Return to NORMAL driving state and clear all stored intersection data."""
        self._in_intersection = False
        self._intersection_entry_time = None
        self._intersection_heading = 0.0
        self._intersection_cte = 0.0
        self._intersection_stable_count = 0

    def _in_cooldown(self, now):
        """True if weak (one_vec / no_vec) triggers are currently suppressed."""
        return now < self._intersect_cooldown_until

    @staticmethod
    def _vector_heading(vector):
        """
        Compute image-space heading of a single edge vector.

        In image coordinates y increases DOWNWARD, so the point LOWER in the image
        (larger y) is closer to the car = "near"; the HIGHER point (smaller y) is
        farther up the road = "far".  This MUST match _aim_x's near/far selection.

        dx = far.x - near.x, dy = far.y - near.y (negative, because far is higher up).

        Returns heading = atan2(dx, -dy):
           0.0         straight up the road (straight ahead in image)
           positive    road curves right (dx > 0)
           negative    road curves left  (dx < 0)
        Range: roughly (-pi/4, +pi/4) for any sane road segment on a forward-facing cam.
        """
        p0, p1 = vector[0], vector[1]
        # Match _aim_x EXACTLY: if p1.y >= p0.y, p1 is LOWER (closer) -> near=p1, far=p0.
        # (This was the bug: we previously had the branches swapped, producing ~140-180deg.)
        if p1.y >= p0.y:
            near, far = p1, p0
        else:
            near, far = p0, p1
        dx = far.x - near.x
        dy = far.y - near.y  # negative (far.y < near.y) when vector points up the road
        # -dy = forward distance in image; atan2 gives the signed angle from "straight up".
        return math.atan2(dx, -dy)

    @staticmethod
    def _lane_heading(vec_left, vec_right):
        """Average heading from both edge vectors to get centerline road heading."""
        return 0.5 * (LineFollower._vector_heading(vec_left)
                      + LineFollower._vector_heading(vec_right))

    @staticmethod
    def _clamp_heading(h, max_deg=45.0):
        """
        Reject headings that are physically implausible for a road in front of the car
        (e.g. the buggy ~140deg values from a swapped near/far, or a near-vertical
        edge from a crosswalk/gate post).  Returns (clamped_heading, ok).
        """
        lim = math.radians(max_deg)
        if abs(h) > lim:
            return max(-lim, min(lim, h)), False
        return h, True

    def _enter_intersection(self, heading, cte, reason):
        """
        Snapshot the lane geometry at the moment we detect we're entering the
        intersection and switch into STRAIGHT_INTERSECTION state.

        We store HEADING and CROSS-TRACK ERROR derived directly from the edge vectors
        (NOT the steering command, NOT the PID output) so we can continue along the
        memorized road direction even when the edge detector goes crazy inside the
        intersection.
        """
        # Double clamp:
        #  1) hard sanity clamp (rejects truly garbage values like cross posts)
        #  2) "straight mission" clamp — STRAIGHT means go nearly straight across,
        #     not carry a 20+ deg approach curve into the intersection.
        lim_hard = math.radians(self.intersect_max_entry_heading_deg)
        lim_lock = math.radians(self.intersect_lock_max_heading_deg)
        clamped_hard = False
        if abs(heading) > lim_hard:
            self.get_logger().warn(
                f"Intersection heading {math.degrees(heading):+.1f}deg implausible; "
                f"clamping to 0 (straight). reason={reason}")
            heading = 0.0
            clamped_hard = True
        elif abs(heading) > lim_lock:
            # Carry only a small remnant of the approach curve; the car should go
            # *mostly* straight through the crossing.
            heading = math.copysign(lim_lock, heading)

        self._in_intersection = True
        self._intersection_entry_time = time.time()
        self._intersection_heading = float(heading)
        self._intersection_cte = float(cte)
        self._intersection_stable_count = 0
        # Reset PID integration so we start clean on exit.
        self.integral = 0.0
        self.prev_time = None
        self.get_logger().info(
            f"*** Entering STRAIGHT_INTERSECTION reason={reason} "
            f"heading={math.degrees(heading):+.1f}deg cte={cte:+.2f}"
            f"{' (clamped)' if clamped_hard else ''}")

    def _exit_intersection(self, reason):
        """Switch back to NORMAL state. Called once stability/safety conditions are met."""
        self.get_logger().info(
            f"*** Exiting STRAIGHT_INTERSECTION reason={reason} "
            f"stable={self._intersection_stable_count}")
        # Clear PID state so recovery doesn't inherit stale integral/derivative from
        # the last NORMAL frame (which may be many frames old).
        self.integral = 0.0
        self.prev_time = None

        # If we exited because of a timeout or a watchdog (i.e. we did NOT recover
        # cleanly via two vectors), the stored heading/cte memory is unreliable.
        # Invalidate it, damp the fallback turn toward 0 (so the "lost" handler goes
        # nearly straight), and set a cooldown during which weak triggers cannot
        # re-enter. Wide/spike triggers (two vectors) can still re-enter because
        # they bring fresh trustworthy geometry.
        if reason in ("timeout", "watchdog"):
            self._last_good_heading = 0.0
            self._last_good_cte = 0.0
            # Damp last_good_turn toward 0 so the lost handler doesn't hold a hard
            # turn from a corrupted dead-reckoning.
            self.last_good_turn = 0.0
            # Force heading EMAs to re-prime from fresh data (the old EMA is probably
            # contaminated by cross-street edges).
            self._heading_ema_init = False
            self._intersect_cooldown_until = time.time() + self.intersect_cooldown_after_timeout

        self._reset_intersection_state()

    # ------------------------------------------------------------------
    # Edge vectors callback
    # ------------------------------------------------------------------
    def edge_vectors_callback(self, message):
        now = time.time()
        img_w = float(message.image_width)
        img_center = img_w / 2.0
        if img_center <= 0:
            return

        count = message.vector_count
        mission_straight = (self.current_mission == "STRAIGHT")

        # ================================================================
        # NEW: Intersection state-machine bookkeeping
        # ================================================================
        # If we're currently in a LEFT/RIGHT mission, intersection mode has no meaning.
        if not mission_straight and self._in_intersection:
            self._reset_intersection_state()

        # --- SAFETY TIMEOUT: if we've been in intersection longer than allowed, bail out ---
        if self._in_intersection and self._intersection_entry_time is not None:
            if (now - self._intersection_entry_time) > self.intersect_max_time:
                self._exit_intersection("timeout")

        # ---------------------------------------------------------------
        # Case A: two vectors seen
        # ---------------------------------------------------------------
        if count >= 2:
            v1, v2 = message.vector_1, message.vector_2
            xa = self._aim_x(v1)
            xb = self._aim_x(v2)
            if self._mean_x(v1) < self._mean_x(v2):
                left_x, right_x = xa, xb
                vec_left, vec_right = v1, v2
            else:
                left_x, right_x = xb, xa
                vec_left, vec_right = v2, v1
            lane_width = right_x - left_x
            # NEW: heading of current two-vector observation (image-space, radians)
            current_heading = self._lane_heading(vec_left, vec_right)

            # ==== STRAIGHT mission with intersection state machine ====
            if mission_straight:
                # --------------------------------------------------------
                # STRAIGHT_INTERSECTION state: two vectors re-appear
                # --------------------------------------------------------
                if self._in_intersection:
                    # Only accept this measurement as "stable" if the lane width looks
                    # normal (i.e. the far cross-street curb is no longer in view, so
                    # we are approaching the exit lane on the other side).
                    width_ok = (lane_width <=
                                self.learned_lane_width * self.intersect_exit_width_ratio
                                and lane_width > 0)
                    if width_ok:
                        self._intersection_stable_count += 1
                        # Width is back to normal -> we are seeing the real exit lane.
                        # Gently blend heading/cte toward the new reading so we don't
                        # snap sideways on exit.
                        blend = self.intersect_heading_blend
                        self._intersection_heading = (
                            (1.0 - blend) * self._intersection_heading
                            + blend * current_heading)
                        new_cte = ((left_x + right_x) * 0.5 - img_center) / img_center
                        self._intersection_cte = (
                            (1.0 - blend) * self._intersection_cte + blend * new_cte)
                    else:
                        # Two vectors but width is still wide -> we are seeing the
                        # cross street edges, NOT the exit lane. DO NOT blend; reset
                        # stability counter and keep driving the stored heading.
                        self._intersection_stable_count = 0

                    # Exit when stable for N consecutive frames with normal width.
                    if self._intersection_stable_count >= self.intersect_stable_frames:
                        self._exit_intersection("recovery")
                        # After exit, re-run the NORMAL-state two-vector logic on THIS SAME
                        # frame's geometry so we don't waste a tick or emit stale commands.
                        # Compute the normal lane center (respecting the very-wide sticky-side
                        # rule, same as NORMAL mode).
                        if lane_width > self.learned_lane_width * self.junction_width_ratio:
                            if self._straight_junction_side is None:
                                cfl = left_x + 0.50 * self.learned_lane_width
                                cfr = right_x - 0.50 * self.learned_lane_width
                                self._straight_junction_side = (
                                    'L' if abs(cfl - img_center) < abs(cfr - img_center) else 'R')
                            if self._straight_junction_side == 'L':
                                lane_center = left_x + 0.50 * self.learned_lane_width
                            else:
                                lane_center = right_x - 0.50 * self.learned_lane_width
                        else:
                            lane_center = (left_x + right_x) / 2.0
                            self._straight_junction_side = None
                        # Update fallbacks / EMA on this recovery frame.
                        if lane_width > 0:
                            if self._width_ema_samples == 0:
                                self._width_ema = lane_width
                            else:
                                self._width_ema = 0.15 * lane_width + 0.85 * self._width_ema
                            self._width_ema_samples += 1
                        self._last_two_vec_time = now
                        self._last_good_heading = current_heading
                        raw_cte = (lane_center - img_center) / img_center
                        self._last_good_cte = raw_cte
                        self.vectors_available = True
                        self.last_vector_time = now
                        if self.learn_lane_width:
                            if 150.0 < lane_width < (img_w * 0.65):
                                self.learned_lane_width = (
                                    0.05 * lane_width + 0.95 * self.learned_lane_width)
                        self.error = max(-1.0, min(1.0, raw_cte))
                        self.target_turn = self._compute_pid(self.error, now)
                        self.target_speed = self._compute_speed(self.target_turn)
                        # Only now, on a clean recovery, update last_good_turn.
                        self.last_good_turn = self.target_turn
                        return

                    # Still inside intersection: drive off STORED heading/cte ONLY.
                    # Ignore noisy observations entirely (cross-street edges, posts).
                    self.vectors_available = True
                    self.last_vector_time = now
                    # Hold CTE constant — no decay, no blending. Decay was causing
                    # lateral drift toward the curb when vectors dropped.
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    # NOTE: do NOT update last_good_turn here. If we timeout, the lost
                    # handler should fall back on the last PRE-INTERsection turn, not
                    # the dead-reckoned intersection turn.
                    # DO NOT update PID state, width learning, width EMA in this mode.
                    return
                # --------------------------------------------------------
                # NORMAL state (straight mission): detect intersection entry
                # --------------------------------------------------------
                else:
                    # Update width EMA and "good heading/cte" memory for spike detection & fallback.
                    self._last_two_vec_time = now
                    if lane_width > 0:
                        if self._width_ema_samples == 0:
                            self._width_ema = lane_width
                        else:
                            self._width_ema = 0.15 * lane_width + 0.85 * self._width_ema
                        self._width_ema_samples += 1

                    # Compute normal center and heading/cte (used both for normal driving
                    # and as the "snapshot" values if we decide to enter intersection now).
                    # We keep the original junction-side sticky logic for very wide crossings
                    # that appear as two vectors (this was already handling some cases before).
                    if lane_width > self.learned_lane_width * self.junction_width_ratio:
                        if self._straight_junction_side is None:
                            center_from_left = left_x + 0.50 * self.learned_lane_width
                            center_from_right = right_x - 0.50 * self.learned_lane_width
                            err_l = abs(center_from_left - img_center)
                            err_r = abs(center_from_right - img_center)
                            self._straight_junction_side = 'L' if err_l < err_r else 'R'
                        if self._straight_junction_side == 'L':
                            lane_center = left_x + 0.50 * self.learned_lane_width
                        else:
                            lane_center = right_x - 0.50 * self.learned_lane_width
                    else:
                        lane_center = (left_x + right_x) / 2.0
                        self._straight_junction_side = None

                    raw_cte = (lane_center - img_center) / img_center

                    # -- Intersection entry triggers (OR-logic, hysteresis via separate entry/exit ratios) --
                    reason = None
                    # Trigger 1: wide lane (far curb of cross street now visible)
                    if lane_width > self.learned_lane_width * self.intersect_entry_width_ratio:
                        reason = "wide"
                    # Trigger 4: sudden width spike vs recent smoothed width (cross street appeared)
                    elif (self._width_ema_samples
                          >= self.intersect_width_samples_for_spike
                          and self._width_ema > 0
                          and lane_width > self._width_ema * self.intersect_width_spike_ratio):
                        reason = "spike"

                    if reason is not None:
                        # Snap heading/cte from the current (still partially trustworthy) geometry.
                        self._last_good_heading = current_heading
                        self._last_good_cte = raw_cte
                        self._enter_intersection(current_heading, raw_cte, reason)
                        # After entering, drive the intersection this frame using stored values.
                        self.vectors_available = True
                        self.last_vector_time = now
                        turn = (self.intersect_heading_gain * self._intersection_heading
                                + self.intersect_cte_gain * self._intersection_cte)
                        self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                        self.target_speed = self.intersect_speed
                        # Snapshot the locked dead-reckoned turn as the "lost" fallback
                        # (we want to KEEP going straight if we timeout).
                        self.last_good_turn = self.target_turn
                        return

                    # --- Normal (non-intersection) straight path ---
                    # Remember these for fallback if we drop vectors.
                    self._last_good_heading = current_heading
                    self._last_good_cte = raw_cte
                    # Update fast & slow heading EMAs.
                    a_fast = self.intersect_fast_ema_alpha
                    a_slow = self.intersect_slow_ema_alpha
                    if not self._heading_ema_init:
                        self._heading_ema_fast = current_heading
                        self._heading_ema_slow = current_heading
                        self._heading_ema_init = True
                    else:
                        self._heading_ema_fast = a_fast * current_heading + (1.0 - a_fast) * self._heading_ema_fast
                        self._heading_ema_slow = a_slow * current_heading + (1.0 - a_slow) * self._heading_ema_slow
                    self.vectors_available = True

                    if self.learn_lane_width:
                        if 150.0 < lane_width < (img_w * 0.65):
                            self.learned_lane_width = (
                                0.05 * lane_width + 0.95 * self.learned_lane_width)

                    self.error = max(-1.0, min(1.0, raw_cte))
                    self.target_turn = self._compute_pid(self.error, now)
                    self.target_speed = self._compute_speed(self.target_turn)
                    self.last_good_turn = self.target_turn
                    return

            # ==== Turn missions (LEFT / RIGHT) — unchanged behavior ====
            else:
                if self.current_mission == "LEFT":
                    offset = self._clamped_offset(0.35 * lane_width, lane_width)
                    lane_center = left_x + offset
                elif self.current_mission == "RIGHT":
                    offset = self._clamped_offset(0.35 * lane_width, lane_width)
                    lane_center = right_x - offset

                self.vectors_available = True
                if self.learn_lane_width:
                    if 150.0 < lane_width < (img_w * 0.65):
                        self.learned_lane_width = (
                            0.05 * lane_width + 0.95 * self.learned_lane_width)

        # ---------------------------------------------------------------
        # Case B: exactly one vector
        # ---------------------------------------------------------------
        elif count == 1:
            v = message.vector_1
            aim = self._aim_x(v)
            mean_x = self._mean_x(v)
            band = self.side_margin * img_center
            if mean_x < img_center - band:
                self.last_single_side = 'LEFT'
            elif mean_x > img_center + band:
                self.last_single_side = 'RIGHT'
            lane_width = self.learned_lane_width
            one_vec_heading = self._vector_heading(v)

            # ---- Straight mission: single-vector handling ----
            if mission_straight:
                # ---- Already in intersection: hold stored heading ----
                if self._in_intersection:
                    # Inside an intersection, a single vector is almost always a cross-street
                    # curb, gate post, or other junk. DO NOT blend its heading; it will yank
                    # the steering into the curb. Just hold the stored heading/cte.
                    self.vectors_available = True
                    self.last_vector_time = now
                    self._intersection_stable_count = 0
                    cte = self._intersection_cte  # hold constant
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    # Do NOT update last_good_turn with dead-reckoned values.
                    return

                # ---- NORMAL mode: does this single-vector frame smell like an intersection? ----
                # In this simulator the dominant intersection cue on a straight road is:
                # the edge detector loses the continuing lane and latches onto the
                # cross-street curb, which has a sharply different heading from the road
                # we were on. Normal curves change heading gradually (EMA'd each frame);
                # a real intersection produces a SUDDEN heading jump vs the fast EMA.
                jump_lim = math.radians(self.intersect_heading_jump_deg)
                have_heading_ref = self._heading_ema_init
                heading_jump = abs(one_vec_heading - self._heading_ema_fast) if have_heading_ref else 0.0
                if (have_heading_ref
                        and not self._in_cooldown(now)
                        and heading_jump > jump_lim):
                    # Detector latched onto a cross-street edge: the edge heading just
                    # jumped sharply away from the fast EMA. Lock to the SLOW EMA
                    # (long-term road direction, i.e. "straight ahead" averaged over the
                    # past ~1 s) so we drive straight across instead of following the
                    # cross street or carrying too much approach-curve bias.
                    snap_heading = self._heading_ema_slow
                    snap_cte = self.error  # use current cte from last NORMAL frame
                    self._enter_intersection(
                        snap_heading, snap_cte,
                        f"one_vec_jump({math.degrees(heading_jump):.0f}deg)")
                    self.vectors_available = True
                    self.last_vector_time = now
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    # Snapshot this locked-straight turn as the fallback for the lost
                    # handler in case we timeout before a clean two-vector recovery.
                    self.last_good_turn = self.target_turn
                    return

            # ---- Original single-vector logic (LEFT/RIGHT missions, or normal straight) ----
            if self.current_mission == "LEFT":
                offset = self._clamped_offset(0.40 * lane_width, lane_width)
            elif self.current_mission == "RIGHT":
                offset = self._clamped_offset(0.60 * lane_width, lane_width)
            else:
                offset = self._clamped_offset(0.50 * lane_width, lane_width)

            if self.last_single_side == 'LEFT':
                lane_center = aim + offset
            else:
                lane_center = aim - (lane_width - offset)

            self.vectors_available = True

        # ---------------------------------------------------------------
        # Case C: no vectors
        # ---------------------------------------------------------------
        else:
            # NEW: If we're already inside a straight intersection and vectors vanish,
            # keep driving the stored heading (no new info to corrupt it).
            if mission_straight and self._in_intersection:
                self.vectors_available = True
                self.last_vector_time = now
                self._intersection_stable_count = 0
                cte = self._intersection_cte
                turn = (self.intersect_heading_gain * self._intersection_heading
                        + self.intersect_cte_gain * cte)
                self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                self.target_speed = self.intersect_speed
                return
            # NOTE: "no vectors at all" does NOT trigger intersection entry.
            # It fires too easily on any detection dropout (gate, shadow, texture),
            # and we have no geometry to snap a heading from. The existing "lost"
            # handler (below in control_loop) uses last_good_turn which is preserved
            # from pre-intersection NORMAL driving.

            # Original "no vector" behavior.
            self.vectors_available = False
            return

        # ---- Common tail for non-intersection single-vector and turn-mission cases ----
        # The single-vector path above sets lane_center but falls through only when not
        # triggering intersection mode.  Update PID from lane_center.
        self.last_vector_time = now
        raw_error = (lane_center - img_center) / img_center
        self.error = max(-1.0, min(1.0, raw_error))
        self.target_turn = self._compute_pid(self.error, now)
        self.target_speed = self._compute_speed(self.target_turn)
        self.last_good_turn = self.target_turn
        # If we're running NORMAL single-vector straight driving, also feed the heading
        # EMAs so our intersection-jump detector has fresh references.
        # Use the single edge's heading — it tracks the road gradually in curves but
        # will SNAP when the detector latches onto a cross-street edge.
        if mission_straight and not self._in_intersection and count == 1:
            a_fast = self.intersect_fast_ema_alpha
            a_slow = self.intersect_slow_ema_alpha
            if not self._heading_ema_init:
                self._heading_ema_fast = one_vec_heading
                self._heading_ema_slow = one_vec_heading
                self._heading_ema_init = True
            else:
                self._heading_ema_fast = a_fast * one_vec_heading + (1.0 - a_fast) * self._heading_ema_fast
                self._heading_ema_slow = a_slow * one_vec_heading + (1.0 - a_slow) * self._heading_ema_slow

    # ------------------------------------------------------------------
    # PID
    # ------------------------------------------------------------------
    def _compute_pid(self, error, now):
        dt = 0.033 if self.prev_time is None else (now - self.prev_time)
        if dt <= 0.0 or dt > 0.5:
            dt = 0.033
        self.prev_time = now

        p = self.Kp * error
        self.integral += error * dt
        self.integral = max(-0.4, min(0.4, self.integral))
        i = self.Ki * self.integral
        d = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        out = p + i + d
        return max(TURN_MIN, min(TURN_MAX, out))

    # ------------------------------------------------------------------
    # Speed shaping
    # ------------------------------------------------------------------
    def _compute_speed(self, turn):
        severity = min(1.0, abs(turn))
        return self.speed_straight - severity * (self.speed_straight - self.speed_sharp)

    # ------------------------------------------------------------------
    # LiDAR / obstacle avoidance (UNCHANGED logic)
    # ------------------------------------------------------------------
    def lidar_callback(self, msg):
        if not self.obstacle_enable:
            self.obstacle_detected = False
            return

        ranges = msg.ranges
        n = len(ranges)
        if n == 0 or msg.angle_increment == 0.0:
            return

        half_fov = math.radians(self.obstacle_fov_deg) / 2.0
        i_center = int(round((0.0 - msg.angle_min) / msg.angle_increment))
        i_half = int(round(half_fov / abs(msg.angle_increment)))

        left_min = float('inf')
        right_min = float('inf')
        for k in range(i_center - i_half, i_center + i_half + 1):
            r = ranges[k % n]
            if not math.isfinite(r) or r <= 0.05:
                continue
            if k >= i_center:
                left_min = min(left_min, r)
            else:
                right_min = min(right_min, r)

        self.nearest_dist = min(left_min, right_min)

        if self.nearest_dist < self.obstacle_trigger_dist:
            self.obstacle_detected = True
        elif self.nearest_dist > self.obstacle_clear_dist:
            self.obstacle_detected = False

        if self.obstacle_detected:
            self.obstacle_turn = (self.obstacle_turn_gain
                                  if right_min > left_min else -self.obstacle_turn_gain)

    # ------------------------------------------------------------------
    # Main control loop (33 Hz)
    # ------------------------------------------------------------------
    def control_loop(self):
        now = time.time()

        # NEW: safety watchdog — if edge_vectors have completely stopped publishing while we're
        # in intersection mode (detector crash / node died), drop out so the existing "lost"
        # handler below can take over.
        if self._in_intersection:
            if self.last_vector_time is not None and (now - self.last_vector_time) > self.intersect_max_time:
                self._exit_intersection("watchdog")

        if self.obstacle_detected:
            # Obstacle avoidance still gets highest priority; it blends with lane/intersection turn.
            want_turn = max(TURN_MIN, min(TURN_MAX,
                            0.35 * self.target_turn + self.obstacle_turn))
            want_speed = self.obstacle_speed

        elif self._in_intersection:
            # NEW: STRAIGHT_INTERSECTION drive — target_turn/target_speed are already
            # set each edge-vector frame from stored heading+cte; if the edge callback
            # hasn't run in a while (e.g. detector stuck), continue using last stored values.
            want_turn = self.target_turn
            want_speed = self.target_speed

        elif self.vectors_available:
            want_turn = self.target_turn
            want_speed = self.target_speed

        else:
            elapsed = 1e9 if self.last_vector_time is None else (now - self.last_vector_time)
            if elapsed <= self.no_vector_hold:
                want_turn = self.last_good_turn
                want_speed = self.speed_lost
            else:
                if self.current_mission == "STRAIGHT":
                    want_turn = self.last_good_turn * 0.5
                else:
                    want_turn = self.last_good_turn
                want_speed = self.speed_lost
            self.integral = 0.0
            self.prev_time = None

        self.filtered_turn = (
            self.steer_alpha * want_turn + (1.0 - self.steer_alpha) * self.filtered_turn)
        self.filtered_speed = (
            self.speed_alpha * want_speed + (1.0 - self.speed_alpha) * self.filtered_speed)

        final_turn = max(TURN_MIN, min(TURN_MAX, self.filtered_turn))
        final_speed = max(SPEED_MIN, min(SPEED_MAX, self.filtered_speed))

        self.publish_drive_cmd(final_speed, self.steer_sign * final_turn)

        self._tick += 1
        if self.debug_log and self._tick % 15 == 0:
            mode = ('INT' if self._in_intersection else
                    ('OBS' if self.obstacle_detected else 'NORM'))
            self.get_logger().info(
                f"vec={'Y' if self.vectors_available else 'N'} "
                f"side={self.last_single_side} "
                f"width={self.learned_lane_width:.0f} "
                f"mission={self.current_mission} "
                f"mode={mode} "
                f"err={self.error:+.3f} "
                f"obs={'Y' if self.obstacle_detected else 'N'}@{self.nearest_dist:.2f} "
                f"turn_int={final_turn:+.3f} "
                f"joy={self.steer_sign * final_turn:+.3f} spd={final_speed:.2f}")

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish_drive_cmd(self, speed, turn):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, float(speed), 0.0, float(turn)]
        self.pub_joy.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
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
