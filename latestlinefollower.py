import math
import time
import statistics
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from synapse_msgs.msg import EdgeVectors
from std_msgs.msg import String, Bool

QOS_PROFILE_DEFAULT = 10
SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0


# =====================================================================
# Mission Finite State Machine
# =====================================================================
#
# States and their meaning:
#
#   NORMAL_LINE_FOLLOWING   — Default.  Pure line following, no zone
#                              detection.  Entered on startup and after
#                              a new /mission/turn restarts the cycle.
#
#   WAITING_FOR_SAFE_ZONE   — /target_qr received.  The buggy continues
#                              line following but the LiDAR safe-zone
#                              detector is now active.  The only way
#                              OUT of this state is a confirmed zone
#                              detection (→ SAFE_ZONE_REACHED) or a
#                              new /mission/turn (→ NORMAL_LINE_FOLLOWING).
#
#   SAFE_ZONE_REACHED       — LiDAR confirmed a safe zone.  The buggy
#                              performs a forward roll (drives straight
#                              at fixed speed for a short time) so it
#                              physically enters the zone, then stops.
#                              Transition: → WAITING_FOR_SERVER_ACK.
#
#   WAITING_FOR_SERVER_ACK  — Buggy is stopped.  The QR Detector is
#                              communicating with the Municipality Server.
#                              Exits on /resume_line_following "RESUME"
#                              (→ NAVIGATING_TO_NEXT_TARGET) or
#                              "MISSION_COMPLETE" (→ MISSION_COMPLETE).
#
#   NAVIGATING_TO_NEXT_TARGET — Server assigned the next target; the
#                              buggy resumes line following.  The next
#                              /target_qr will push it back into
#                              WAITING_FOR_SAFE_ZONE.
#
#   MISSION_COMPLETE        — All deliveries done.  Stopped.  A new
#                              /mission/turn restarts the cycle
#                              (→ NORMAL_LINE_FOLLOWING).
#
# =====================================================================
class MissionState:
    NORMAL_LINE_FOLLOWING    = "NORMAL_LINE_FOLLOWING"
    WAITING_FOR_SAFE_ZONE    = "WAITING_FOR_SAFE_ZONE"
    SAFE_ZONE_REACHED        = "SAFE_ZONE_REACHED"
    WAITING_FOR_SERVER_ACK   = "WAITING_FOR_SERVER_ACK"
    NAVIGATING_TO_NEXT_TARGET = "NAVIGATING_TO_NEXT_TARGET"
    MISSION_COMPLETE         = "MISSION_COMPLETE"


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
        self.declare_parameter('speed_straight', 0.65)
        self.declare_parameter('speed_sharp', 0.35)
        self.declare_parameter('speed_lost', 0.30)
        self.declare_parameter('steer_alpha', 0.55)
        self.declare_parameter('speed_alpha', 0.15)
        self.declare_parameter('no_vector_hold', 0.6)
        self.declare_parameter('debug_log', True)
        self.declare_parameter('obstacle_enable', True)
        self.declare_parameter('obstacle_trigger_dist', 0.90)
        self.declare_parameter('obstacle_clear_dist', 1.30)
        self.declare_parameter('obstacle_fov_deg', 60.0)
        self.declare_parameter('obstacle_turn_gain', 0.9)
        self.declare_parameter('obstacle_speed', 0.32)

        # --- edge-safety margin so the aim point never sits right on the curb ---
        self.declare_parameter('turn_edge_margin_px', 35.0)

        # --- junction lane-width ratio that triggers "wide crossing" handling ---
        self.declare_parameter('junction_width_ratio', 1.8)

        # Straight-intersection state-machine parameters
        self.declare_parameter('intersect_entry_width_ratio', 1.6)
        self.declare_parameter('intersect_exit_width_ratio', 1.25)
        self.declare_parameter('intersect_width_spike_ratio', 1.35)
        self.declare_parameter('intersect_stable_frames', 6)
        self.declare_parameter('intersect_max_time', 4.0)
        self.declare_parameter('intersect_no_vec_time', 0.18)
        self.declare_parameter('intersect_heading_gain', 0.30)
        self.declare_parameter('intersect_cte_gain', 0.20)
        self.declare_parameter('intersect_heading_blend', 0.15)
        self.declare_parameter('intersect_speed', 0.55)
        self.declare_parameter('intersect_width_samples_for_spike', 6)
        self.declare_parameter('intersect_one_vec_time', 0.25)
        self.declare_parameter('intersect_cooldown_after_timeout', 1.5)
        self.declare_parameter('intersect_max_entry_heading_deg', 30.0)
        self.declare_parameter('intersect_heading_jump_deg', 20.0)
        self.declare_parameter('intersect_memory_max_age', 1.0)
        self.declare_parameter('intersect_lock_max_heading_deg', 8.0)
        self.declare_parameter('intersect_slow_ema_alpha', 0.07)
        self.declare_parameter('intersect_fast_ema_alpha', 0.4)

        # =====================================================================
        # Safe Zone detection parameters (competition-grade)
        # =====================================================================
        #
        # zone_enter_distance  — Percentile distance must drop BELOW this
        #   to start counting toward confirmation.  This is the "trigger"
        #   threshold.  (0.90 m)
        #
        # zone_exit_distance   — Percentile distance must rise ABOVE this
        #   to declare the frame a "miss" and increment the miss counter.
        #   Must be > zone_enter_distance to create hysteresis, which
        #   prevents the detector from flickering at the boundary.  (1.40 m)
        #
        # zone_confirm_count   — How many *successful* frames (not
        #   necessarily consecutive — see tolerance) are required before
        #   the zone is declared confirmed.  (5)
        #
        # zone_confirm_tolerance — How many *missed* frames are allowed
        #   within the current confirmation window before the confirm
        #   counter is reset.  This prevents a single bad LiDAR frame
        #   from discarding all accumulated evidence.  (2)
        #
        # zone_percentile      — Which percentile of valid sector
        #   distances to use as the representative measurement.  The
        #   25th percentile is robust to a few outlier short readings
        #   (sensor noise / ground bounce) and a few inf readings
        #   (missed returns), while still being responsive to a genuine
        #   nearby wall.  (25)
        #
        # zone_min_valid_points — Minimum number of valid LiDAR returns
        #   in the sector for the frame to be considered at all.  If
        #   fewer valid points exist, the frame is discarded (neither
        #   a hit nor a miss).  This prevents false positives from
        #   sparse data.  (3)
        #
        # zone_fov_min_deg / zone_fov_max_deg — Asymmetric sector that
        #   covers the building face as the buggy approaches.  (-45° to
        #   +15°)
        #
        # forward_roll_time    — Duration of the forward roll after safe
        #   zone confirmation, so the buggy physically enters the zone.
        #   (0.80 s)
        #
        self.declare_parameter('zone_enter_distance', 0.90)
        self.declare_parameter('zone_exit_distance', 1.40)
        self.declare_parameter('zone_confirm_count', 5)
        self.declare_parameter('zone_confirm_tolerance', 2)
        self.declare_parameter('zone_percentile', 25)
        self.declare_parameter('zone_min_valid_points', 3)
        self.declare_parameter('forward_roll_time', 0.80)
        self.declare_parameter('zone_fov_min_deg', -45.0)
        self.declare_parameter('zone_fov_max_deg', 15.0)

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

        # Straight-intersection state machine runtime state
        self._in_intersection = False
        self._intersection_entry_time = None
        self._intersection_heading = 0.0
        self._intersection_cte = 0.0
        self._intersection_stable_count = 0
        self._width_ema = self.lane_width_px
        self._width_ema_samples = 0
        self._last_good_heading = 0.0
        self._last_good_cte = 0.0
        self._heading_ema_fast = 0.0
        self._heading_ema_slow = 0.0
        self._heading_ema_init = False
        self._last_two_vec_time = None
        self._intersect_cooldown_until = 0.0

        # =====================================================================
        # Mission Finite State Machine
        # =====================================================================
        self.mission_state = MissionState.NORMAL_LINE_FOLLOWING
        self.target_qr_string = ""          # stored for debug logging only
        self.last_valid_mission = "NONE"

        # Safe-zone detection bookkeeping (used only in WAITING_FOR_SAFE_ZONE)
        self._zone_confirm_counter = 0      # successful frames accumulated
        self._zone_miss_counter = 0         # consecutive miss frames
        self._safe_zone_published = False   # True once /safe_zone has been
                                             # published for this target —
                                             # prevents duplicate publications
        self._forward_roll_start_time = None

        # LiDAR zone analysis results (written by lidar_callback, read by
        # control_loop).  All inf/NaN are filtered out before these are
        # computed.
        self._zone_valid_distances = []     # list of valid distances this frame
        self._zone_percentile_dist = float('inf')  # representative distance
        self._zone_sector_data = []         # raw sector data for debug logging

        # Timers for periodic log messages
        self._wait_log_time = 0.0
        self._lidar_debug_time = 0.0
        self._mission_complete_log_time = 0.0

        # ---------------- ROS plumbing ----------------
        self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/mission/turn', self.mission_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/target_qr', self.target_qr_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/resume_line_following', self.resume_callback, 10)

        self.pub_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.pub_safe_zone = self.create_publisher(Bool, '/safe_zone', QOS_PROFILE_DEFAULT)

        self.create_timer(0.033, self.control_loop)
        self.create_timer(1.0, self._reload_params)

        self.get_logger().info(
            "Lane-following controller loaded.\n"
            f"Mission FSM initial state: {self.mission_state}")

    # =====================================================================
    # Mission state transition helper
    # =====================================================================
    def _transition_mission_state(self, new_state, reason=""):
        """Log and execute a mission state transition."""
        old = self.mission_state
        if old == new_state:
            return
        self.mission_state = new_state
        self.get_logger().info(
            f"MISSION STATE: {old} → {new_state}"
            f"  ({reason})" if reason else "")

    # =====================================================================
    # Safe-zone detection reset
    # =====================================================================
    def _reset_zone_detection(self):
        """Clear all safe-zone detection counters so a fresh cycle can
        begin when the next /target_qr arrives."""
        self._zone_confirm_counter = 0
        self._zone_miss_counter = 0
        self._safe_zone_published = False
        self._zone_percentile_dist = float('inf')
        self._zone_valid_distances = []

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

        # Intersection parameters
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

        # Safe Zone parameters (competition-grade)
        self.zone_enter_distance = g('zone_enter_distance')
        self.zone_exit_distance = g('zone_exit_distance')
        self.zone_confirm_count = int(g('zone_confirm_count'))
        self.zone_confirm_tolerance = int(g('zone_confirm_tolerance'))
        self.zone_percentile = int(g('zone_percentile'))
        self.zone_min_valid_points = int(g('zone_min_valid_points'))
        self.forward_roll_time = g('forward_roll_time')
        self.zone_fov_min_deg = g('zone_fov_min_deg')
        self.zone_fov_max_deg = g('zone_fov_max_deg')

    # ------------------------------------------------------------------
    # Target QR callback
    # ------------------------------------------------------------------
    def target_qr_callback(self, msg):
        """Receive target QR availability from the QR Detector.

        The Line Follower does NOT know whether the target is a patient,
        hospital, or any future mission type.  It simply transitions to
        WAITING_FOR_SAFE_ZONE so the LiDAR detector becomes active.
        """
        if not msg.data or not msg.data.strip():
            return

        self.target_qr_string = msg.data.strip()

        # Transition to WAITING_FOR_SAFE_ZONE from any state that is
        # actively line-following (not stopped).
        if self.mission_state in (
            MissionState.NORMAL_LINE_FOLLOWING,
            MissionState.NAVIGATING_TO_NEXT_TARGET,
        ):
            self._reset_zone_detection()
            self._transition_mission_state(
                MissionState.WAITING_FOR_SAFE_ZONE,
                f"/target_qr received: {self.target_qr_string}")
        elif self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            # Already waiting — a new /target_qr may arrive if the QR
            # Detector re-publishes.  Reset the detection cycle so we
            # start fresh.
            self._reset_zone_detection()
            self.get_logger().info(
                "/target_qr re-received while WAITING_FOR_SAFE_ZONE; "
                "zone detection reset.")
        else:
            self.get_logger().info(
                f"/target_qr ignored in state {self.mission_state}")

    # ------------------------------------------------------------------
    # Resume Line Following callback
    # ------------------------------------------------------------------
    def resume_callback(self, msg):
        """Receive /resume_line_following from QR Detector.

        Two valid payloads:
          "RESUME"          — server assigned the next target; resume
                               line following.
          "MISSION_COMPLETE" — all deliveries done; enter MISSION_COMPLETE.
        """
        received = msg.data.strip() if msg.data else ""

        if received == "MISSION_COMPLETE":
            if self.mission_state != MissionState.WAITING_FOR_SERVER_ACK:
                self.get_logger().info(
                    f"MISSION_COMPLETE ignored in state {self.mission_state}")
                return
            self._transition_mission_state(
                MissionState.MISSION_COMPLETE,
                "Server signalled mission complete")
            self._mission_complete_log_time = 0.0
            return

        if received != "RESUME":
            self.get_logger().info(f"Ignoring resume message: {received}")
            return

        if self.mission_state != MissionState.WAITING_FOR_SERVER_ACK:
            self.get_logger().info(
                f"RESUME ignored in state {self.mission_state}")
            return

        self._transition_mission_state(
            MissionState.NAVIGATING_TO_NEXT_TARGET,
            "Server ACK / RESUME received")
        self._wait_log_time = 0.0
        # Reset zone detection state so the buggy cannot accidentally
        # perform detection again.  A new /target_qr is required.
        self._reset_zone_detection()
        self.target_qr_string = ""

    # ------------------------------------------------------------------
    # Mission topic callback
    # ------------------------------------------------------------------
    def mission_callback(self, msg):
        mission = msg.data

        # --- Handle NONE / empty / whitespace ---
        if not mission or not mission.strip() or mission.strip() == "NONE":
            self.get_logger().info("Mission Topic Received: NONE — ignoring")
            return

        mission = mission.strip()

        if mission not in ["LEFT", "RIGHT", "STRAIGHT"]:
            return

        self.get_logger().info(f"Mission Topic Received: {mission}")

        # --- MISSION_COMPLETE: exit on new mission ---
        if self.mission_state == MissionState.MISSION_COMPLETE:
            self.get_logger().info(
                "====================================\n"
                "New Mission Received\n"
                "Restarting Navigation\n"
                "====================================")
            self._reset_zone_detection()
            self.target_qr_string = ""
            self._transition_mission_state(
                MissionState.NORMAL_LINE_FOLLOWING,
                "New mission after MISSION_COMPLETE")
            self.current_mission = mission
            self.last_valid_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False
            return

        # --- WAITING_FOR_SERVER_ACK: accept new mission direction ---
        if self.mission_state == MissionState.WAITING_FOR_SERVER_ACK:
            if mission == self.last_valid_mission:
                self.get_logger().info("Ignoring duplicate mission")
                return

            self.get_logger().info(
                "Assignment Accepted — Resuming Navigation")
            self._reset_zone_detection()
            self.target_qr_string = ""
            self._transition_mission_state(
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                "New mission direction while waiting for server")
            self.last_valid_mission = mission
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False
            return

        # --- WAITING_FOR_SAFE_ZONE: new mission may change direction ---
        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            if mission != self.current_mission:
                self.get_logger().info(
                    f"Mission changed to {mission} while WAITING_FOR_SAFE_ZONE")
                self.current_mission = mission
                self._straight_junction_side = None
                self._reset_intersection_state()
                self._heading_ema_init = False
            self.last_valid_mission = mission
            return

        # --- Normal (line-following) mission update ---
        if mission != self.current_mission:
            self.get_logger().info(f"Mission changed to {mission}")
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False

        self.last_valid_mission = mission

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

    # ------------------------------------------------------------------
    # Intersection state-machine helpers
    # ------------------------------------------------------------------
    def _reset_intersection_state(self):
        self._in_intersection = False
        self._intersection_entry_time = None
        self._intersection_heading = 0.0
        self._intersection_cte = 0.0
        self._intersection_stable_count = 0

    def _in_cooldown(self, now):
        return now < self._intersect_cooldown_until

    @staticmethod
    def _vector_heading(vector):
        p0, p1 = vector[0], vector[1]
        if p1.y >= p0.y:
            near, far = p1, p0
        else:
            near, far = p0, p1
        dx = far.x - near.x
        dy = far.y - near.y
        return math.atan2(dx, -dy)

    @staticmethod
    def _lane_heading(vec_left, vec_right):
        return 0.5 * (LineFollower._vector_heading(vec_left)
                      + LineFollower._vector_heading(vec_right))

    @staticmethod
    def _clamp_heading(h, max_deg=45.0):
        lim = math.radians(max_deg)
        if abs(h) > lim:
            return max(-lim, min(lim, h)), False
        return h, True

    def _enter_intersection(self, heading, cte, reason):
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
            heading = math.copysign(lim_lock, heading)

        self._in_intersection = True
        self._intersection_entry_time = time.time()
        self._intersection_heading = float(heading)
        self._intersection_cte = float(cte)
        self._intersection_stable_count = 0

        self.integral = 0.0
        self.prev_time = None

        self.get_logger().info(
            f"*** Entering STRAIGHT_INTERSECTION reason={reason} "
            f"heading={math.degrees(heading):+.1f}deg cte={cte:+.2f}"
            f"{' (clamped)' if clamped_hard else ''}")

    def _exit_intersection(self, reason):
        self.get_logger().info(
            f"*** Exiting STRAIGHT_INTERSECTION reason={reason} "
            f"stable={self._intersection_stable_count}")

        self.integral = 0.0
        self.prev_time = None

        if reason in ("timeout", "watchdog"):
            self._last_good_heading = 0.0
            self._last_good_cte = 0.0
            self.last_good_turn = 0.0
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

        if not mission_straight and self._in_intersection:
            self._reset_intersection_state()

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
            current_heading = self._lane_heading(vec_left, vec_right)

            if mission_straight:

                if self._in_intersection:
                    width_ok = (lane_width <=
                                self.learned_lane_width * self.intersect_exit_width_ratio
                                and lane_width > 0)

                    if width_ok:
                        self._intersection_stable_count += 1
                        blend = self.intersect_heading_blend
                        self._intersection_heading = (
                            (1.0 - blend) * self._intersection_heading
                            + blend * current_heading)
                        new_cte = ((left_x + right_x) * 0.5 - img_center) / img_center
                        self._intersection_cte = (
                            (1.0 - blend) * self._intersection_cte + blend * new_cte)
                    else:
                        self._intersection_stable_count = 0

                    if self._intersection_stable_count >= self.intersect_stable_frames:
                        self._exit_intersection("recovery")

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
                        self.last_good_turn = self.target_turn
                        return

                    self.vectors_available = True
                    self.last_vector_time = now
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    return

                else:
                    self._last_two_vec_time = now
                    if lane_width > 0:
                        if self._width_ema_samples == 0:
                            self._width_ema = lane_width
                        else:
                            self._width_ema = 0.15 * lane_width + 0.85 * self._width_ema
                        self._width_ema_samples += 1

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
                    reason = None

                    if lane_width > self.learned_lane_width * self.intersect_entry_width_ratio:
                        reason = "wide"
                    elif (self._width_ema_samples
                          >= self.intersect_width_samples_for_spike
                          and self._width_ema > 0
                          and lane_width > self._width_ema * self.intersect_width_spike_ratio):
                        reason = "spike"

                    if reason is not None:
                        self._last_good_heading = current_heading
                        self._last_good_cte = raw_cte
                        self._enter_intersection(current_heading, raw_cte, reason)

                        self.vectors_available = True
                        self.last_vector_time = now
                        turn = (self.intersect_heading_gain * self._intersection_heading
                                + self.intersect_cte_gain * self._intersection_cte)
                        self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                        self.target_speed = self.intersect_speed
                        self.last_good_turn = self.target_turn
                        return

                    self._last_good_heading = current_heading
                    self._last_good_cte = raw_cte

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

            if mission_straight:
                if self._in_intersection:
                    self.vectors_available = True
                    self.last_vector_time = now
                    self._intersection_stable_count = 0
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    return

                jump_lim = math.radians(self.intersect_heading_jump_deg)
                have_heading_ref = self._heading_ema_init
                heading_jump = abs(one_vec_heading - self._heading_ema_fast) if have_heading_ref else 0.0

                if (have_heading_ref
                        and not self._in_cooldown(now)
                        and heading_jump > jump_lim):
                    snap_heading = self._heading_ema_slow
                    snap_cte = self.error
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
                    self.last_good_turn = self.target_turn
                    return

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

            self.vectors_available = False
            return

        # ---- Common tail for non-intersection single-vector and turn-mission cases ----
        self.last_vector_time = now
        raw_error = (lane_center - img_center) / img_center
        self.error = max(-1.0, min(1.0, raw_error))
        self.target_turn = self._compute_pid(self.error, now)
        self.target_speed = self._compute_speed(self.target_turn)
        self.last_good_turn = self.target_turn

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
    # LiDAR callback — obstacle avoidance + safe-zone sector analysis
    # ------------------------------------------------------------------
    def lidar_callback(self, msg):
        if not self.obstacle_enable:
            self.obstacle_detected = False
            return

        ranges = msg.ranges
        n = len(ranges)
        if n == 0 or msg.angle_increment == 0.0:
            return

        # ---- Obstacle avoidance sector (symmetric ±FOV/2 around 0°) ----
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

        # =====================================================================
        # Safe Zone LiDAR sector (asymmetric, -45° to +15°)
        # =====================================================================
        # This sector is always computed regardless of mission state so
        # that the data is fresh when the FSM enters WAITING_FOR_SAFE_ZONE.
        # Only the control_loop decides whether to *act* on it.
        p_min_deg = self.zone_fov_min_deg
        p_max_deg = self.zone_fov_max_deg
        i_p_min = int(round((math.radians(p_min_deg) - msg.angle_min) / msg.angle_increment))
        i_p_max = int(round((math.radians(p_max_deg) - msg.angle_min) / msg.angle_increment))
        i_p_min = max(0, min(n - 1, i_p_min))
        i_p_max = max(0, min(n - 1, i_p_max))

        # Collect raw sector data (for debug logging) and valid distances.
        valid_distances = []
        sector_data = []
        for k in range(i_p_min, i_p_max + 1):
            idx = k % n
            r = ranges[idx]
            angle_deg = math.degrees(msg.angle_min + idx * msg.angle_increment)
            sector_data.append((idx, r, angle_deg))
            # WHY: Reject inf (no return), NaN (corrupt), and <= 0.05 m
            # (sensor artifact / ground bounce).  These are not real walls.
            if math.isfinite(r) and r > 0.05:
                valid_distances.append(r)

        self._zone_sector_data = sector_data
        self._zone_valid_distances = valid_distances

        # WHY percentile instead of minimum: a single outlier short
        # reading (noise, ground bounce) can make the minimum
        # misleadingly low.  The 25th percentile is robust to a few
        # outlier short readings while still being responsive to a
        # genuine nearby wall.
        if len(valid_distances) >= self.zone_min_valid_points:
            sorted_d = sorted(valid_distances)
            # Compute the percentile index.
            # WHY: Using the nearest-rank method so the result is
            # deterministic and does not depend on numpy.
            rank = (self.zone_percentile / 100.0) * (len(sorted_d) - 1)
            lower = int(math.floor(rank))
            upper = min(lower + 1, len(sorted_d) - 1)
            frac = rank - lower
            self._zone_percentile_dist = (
                sorted_d[lower] * (1.0 - frac) + sorted_d[upper] * frac)
        else:
            # WHY: Not enough valid points to trust the measurement.
            # Set to inf so the frame is treated as "no evidence" —
            # neither a hit nor a miss — and the miss counter is NOT
            # incremented.
            self._zone_percentile_dist = float('inf')

    # ------------------------------------------------------------------
    # Main control loop (33 Hz)
    # ------------------------------------------------------------------
    def control_loop(self):
        now = time.time()

        # =================================================================
        # MISSION_COMPLETE — highest priority, buggy stopped
        # =================================================================
        if self.mission_state == MissionState.MISSION_COMPLETE:
            self.publish_drive_cmd(0.0, 0.0)
            if now - self._mission_complete_log_time >= 1.0:
                self._mission_complete_log_time = now
                self.get_logger().info(
                    f"[{self.mission_state}] Waiting For New Mission...")
            return

        # =================================================================
        # WAITING_FOR_SERVER_ACK — buggy stopped, waiting for QR Detector
        # =================================================================
        if self.mission_state == MissionState.WAITING_FOR_SERVER_ACK:
            self.publish_drive_cmd(0.0, 0.0)
            if now - self._wait_log_time >= 1.0:
                self._wait_log_time = now
                self.get_logger().info(
                    f"[{self.mission_state}] "
                    "Waiting for Server Assignment...")
            return

        # =================================================================
        # SAFE_ZONE_REACHED — forward roll at fixed speed
        # =================================================================
        if self.mission_state == MissionState.SAFE_ZONE_REACHED:
            if self._forward_roll_start_time is None:
                # Should not happen, but guard against it.
                self._transition_mission_state(
                    MissionState.WAITING_FOR_SERVER_ACK,
                    "Forward roll start time missing")
                return

            elapsed_roll = now - self._forward_roll_start_time
            if elapsed_roll >= self.forward_roll_time:
                # Roll time expired — transition to WAITING_FOR_SERVER_ACK.
                self._forward_roll_start_time = None
                self._transition_mission_state(
                    MissionState.WAITING_FOR_SERVER_ACK,
                    "Forward roll finished")
            else:
                # Still rolling — drive straight at fixed speed.
                self.publish_drive_cmd(0.45, 0.0)
                if int(elapsed_roll * 10) % 5 == 0:
                    self.get_logger().info(
                        f"[{self.mission_state}] FORWARD_ROLL "
                        f"elapsed={elapsed_roll:.3f}s")
            return

        # =================================================================
        # WAITING_FOR_SAFE_ZONE — LiDAR zone detection active
        # =================================================================
        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            self._run_safe_zone_detector(now)

        # =================================================================
        # Intersection watchdog
        # =================================================================
        if self._in_intersection:
            if self.last_vector_time is not None and (now - self.last_vector_time) > self.intersect_max_time:
                self._exit_intersection("watchdog")

        # =================================================================
        # Driving logic (unchanged)
        # =================================================================
        if self.obstacle_detected:
            want_turn = max(TURN_MIN, min(TURN_MAX,
                            0.35 * self.target_turn + self.obstacle_turn))
            want_speed = self.obstacle_speed

        elif self._in_intersection:
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
                f"fsm={self.mission_state} "
                f"mode={mode} "
                f"err={self.error:+.3f} "
                f"obs={'Y' if self.obstacle_detected else 'N'}@{self.nearest_dist:.2f} "
                f"turn_int={final_turn:+.3f} "
                f"joy={self.steer_sign * final_turn:+.3f} spd={final_speed:.2f}")

    # =====================================================================
    # Robust Safe Zone Detector
    # =====================================================================
    def _run_safe_zone_detector(self, now):
        """Competition-grade safe-zone detection.

        Called once per control-loop tick ONLY when the FSM is in
        WAITING_FOR_SAFE_ZONE.  Uses the LiDAR sector analysis from
        lidar_callback.

        Algorithm (each condition explained inline):

        1. Minimum valid points check — if the sector has too few
           valid returns, the frame is discarded entirely.  This
           prevents false positives from sparse/noisy data.

        2. Percentile-based distance — the 25th percentile of valid
           distances in the sector is used instead of the minimum.
           Rationale: a single outlier short reading (noise, ground
           bounce) can make the minimum misleadingly low, while a
           single inf reading can make it misleadingly high.  The
           percentile is robust to both.

        3. Hysteresis — there are two thresholds:
             zone_enter_distance (0.90 m): percentile must drop below
               this to count as a "hit" frame.
             zone_exit_distance (1.40 m): percentile must rise above
               this to count as a "miss" frame.
           If the percentile is between the two thresholds, the frame
           is "indeterminate" — neither a hit nor a miss.  This
           prevents the detector from flickering at the boundary.

        4. Consecutive-frame confirmation with tolerance —
           zone_confirm_count (5) successful frames are required to
           confirm the zone.  However, up to zone_confirm_tolerance
           (2) miss frames are allowed before the confirm counter is
           reset.  This prevents a single bad LiDAR frame from
           discarding all accumulated evidence.

        5. Duplicate publication prevention — the _safe_zone_published
           flag ensures /safe_zone is published exactly ONCE per target.
        """
        valid_count = len(self._zone_valid_distances)
        p_dist = self._zone_percentile_dist

        # ---- Periodic debug logging (every 0.5 s) ----
        if now - self._lidar_debug_time >= 0.5:
            self._lidar_debug_time = now
            self._log_zone_debug(now)

        # ---- Frame classification ----
        # WHY minimum valid points: if the sector has too few returns,
        # the percentile is unreliable.  Treat as "no evidence" — do
        # not increment hit or miss counters.
        if valid_count < self.zone_min_valid_points:
            # Sparse data — skip this frame entirely.
            self.get_logger().debug(
                f"[SafeZone] Sparse frame: {valid_count} valid pts "
                f"(need {self.zone_min_valid_points})")
            return

        # WHY hysteresis: three zones — enter, indeterminate, exit.
        # - p_dist < enter  → HIT  (increment confirm counter)
        # - p_dist > exit   → MISS (increment miss counter)
        # - enter ≤ p_dist ≤ exit → INDETERMINATE (no change)
        if p_dist < self.zone_enter_distance:
            # ---- HIT frame ----
            self._zone_confirm_counter += 1
            self._zone_miss_counter = 0  # reset miss streak

            self.get_logger().info(
                f"[SafeZone] HIT  p_dist={p_dist:.2f}m < "
                f"enter={self.zone_enter_distance:.2f}m  "
                f"confirm={self._zone_confirm_counter}/{self.zone_confirm_count}  "
                f"valid={valid_count}")

            if self._zone_confirm_counter >= self.zone_confirm_count:
                self._on_safe_zone_confirmed(p_dist)

        elif p_dist > self.zone_exit_distance:
            # ---- MISS frame ----
            self._zone_miss_counter += 1

            self.get_logger().info(
                f"[SafeZone] MISS p_dist={p_dist:.2f}m > "
                f"exit={self.zone_exit_distance:.2f}m  "
                f"miss_streak={self._zone_miss_counter}/{self.zone_confirm_tolerance}  "
                f"confirm={self._zone_confirm_counter}/{self.zone_confirm_count}")

            # WHY tolerance: allow a few misses before resetting.  If
            # the miss streak exceeds the tolerance, all accumulated
            # evidence is discarded.
            if self._zone_miss_counter > self.zone_confirm_tolerance:
                if self._zone_confirm_counter > 0:
                    self.get_logger().info(
                        f"[SafeZone] Counter RESET — "
                        f"miss streak ({self._zone_miss_counter}) > "
                        f"tolerance ({self.zone_confirm_tolerance})")
                self._zone_confirm_counter = 0
                self._zone_miss_counter = 0

        else:
            # ---- INDETERMINATE frame ----
            # WHY: The percentile is between enter and exit thresholds.
            # This is the hysteresis band.  Neither confirm nor reset.
            self.get_logger().info(
                f"[SafeZone] ???  p_dist={p_dist:.2f}m  "
                f"(enter={self.zone_enter_distance:.2f}m "
                f"exit={self.zone_exit_distance:.2f}m)  "
                f"confirm={self._zone_confirm_counter}/{self.zone_confirm_count}")

    # =====================================================================
    # Safe Zone confirmed — publish and transition
    # =====================================================================
    def _on_safe_zone_confirmed(self, p_dist):
        """Called exactly once per target when the zone is confirmed."""
        # WHY duplicate prevention: the _safe_zone_published flag
        # ensures /safe_zone is published exactly ONCE even if
        # _run_safe_zone_detector is called again before the state
        # transition takes effect.
        if self._safe_zone_published:
            self.get_logger().warn(
                "[SafeZone] Already published for this target — ignoring.")
            return

        self._safe_zone_published = True
        self._zone_confirm_counter = 0
        self._zone_miss_counter = 0

        # Publish /safe_zone
        zone_msg = Bool()
        zone_msg.data = True
        self.pub_safe_zone.publish(zone_msg)

        self.get_logger().info(
            "========================================\n"
            "SAFE ZONE DETECTED\n"
            f"Percentile distance: {p_dist:.2f}m\n"
            f"Confirm count: {self.zone_confirm_count}\n"
            "Publishing /safe_zone\n"
            "========================================")

        # Transition to SAFE_ZONE_REACHED (forward roll).
        self._forward_roll_start_time = time.time()
        self._transition_mission_state(
            MissionState.SAFE_ZONE_REACHED,
            f"Safe zone confirmed at {p_dist:.2f}m")

    # =====================================================================
    # LiDAR debug logging for safe-zone sector
    # =====================================================================
    def _log_zone_debug(self, now):
        """Print detailed LiDAR sector analysis for debugging."""
        sector = self._zone_sector_data
        if not sector:
            return

        valid = [(idx, r, ang) for idx, r, ang in sector
                 if math.isfinite(r) and r > 0.05]
        valid_count = len(valid)
        total_count = len(sector)

        if valid_count > 0:
            dists = sorted(r for _, r, _ in valid)
            min_dist = dists[0]
            max_dist = dists[-1]
            avg_dist = sum(dists) / len(dists)
            # Compute median
            mid = len(dists) // 2
            if len(dists) % 2 == 0:
                median_dist = (dists[mid - 1] + dists[mid]) / 2.0
            else:
                median_dist = dists[mid]
        else:
            min_dist = float('inf')
            max_dist = float('inf')
            avg_dist = float('inf')
            median_dist = float('inf')

        def fmt(r):
            if not math.isfinite(r):
                return 'inf'
            return f'{r:.2f}'

        self.get_logger().info(
            "================ LIDAR DEBUG ================\n"
            f"Sector       : {self.zone_fov_min_deg:.0f}° to {self.zone_fov_max_deg:.0f}°\n"
            f"Total points : {total_count}\n"
            f"Valid points : {valid_count}  "
            f"(min required: {self.zone_min_valid_points})\n"
            f"Min distance : {fmt(min_dist)}\n"
            f"Median dist  : {fmt(median_dist)}\n"
            f"Avg distance : {fmt(avg_dist)}\n"
            f"Max distance : {fmt(max_dist)}\n"
            f"Percentile   : P{self.zone_percentile} = {fmt(self._zone_percentile_dist)}\n"
            f"Enter thresh : {self.zone_enter_distance:.2f}m\n"
            f"Exit thresh  : {self.zone_exit_distance:.2f}m\n"
            f"Hysteresis   : {self.zone_exit_distance - self.zone_enter_distance:.2f}m\n"
            f"Confirm      : {self._zone_confirm_counter}/{self.zone_confirm_count}  "
            f"(tolerance: {self.zone_confirm_tolerance})\n"
            f"Miss streak  : {self._zone_miss_counter}\n"
            f"FSM state    : {self.mission_state}\n"
            "============================================")

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
