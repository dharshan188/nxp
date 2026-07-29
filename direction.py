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

        self.get_logger().info("Minimal lane-following controller loaded.")

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

    def mission_callback(self, msg):
        mission = msg.data
        if mission in ["LEFT", "RIGHT", "STRAIGHT"]:
            if mission != self.current_mission:
                self.get_logger().info(f"Mission changed to {mission}")
                self.current_mission = mission

    def _aim_x(self, vector):
        p0, p1 = vector[0], vector[1]
        near, far = (p1, p0) if p1.y >= p0.y else (p0, p1)
        b = self.lookahead_blend
        return (1.0 - b) * near.x + b * far.x

    @staticmethod
    def _mean_x(vector):
        return (vector[0].x + vector[1].x) / 2.0

    def edge_vectors_callback(self, message):
        now = time.time()
        img_w = float(message.image_width)
        img_center = img_w / 2.0
        if img_center <= 0:
            return

        count = message.vector_count
        if count >= 2:
            v1, v2 = message.vector_1, message.vector_2
            xa = self._aim_x(v1)
            xb = self._aim_x(v2)
            
            if self._mean_x(v1) < self._mean_x(v2):
                left_x, right_x = xa, xb
            else:
                left_x, right_x = xb, xa
                
            lane_width = right_x - left_x
            
            if self.current_mission == "STRAIGHT":
                # Only use "straighter line" logic if we are 100% sure we are at a junction
                if lane_width > self.learned_lane_width * 1.8:
                    center_from_left = left_x + 0.50 * self.learned_lane_width
                    center_from_right = right_x - 0.50 * self.learned_lane_width
                    err_l = abs(center_from_left - img_center)
                    err_r = abs(center_from_right - img_center)
                    lane_center = center_from_left if err_l < err_r else center_from_right
                else:
                    # Normal road: Always use standard averaging for straight missions
                    lane_center = (left_x + right_x) / 2.0
            else:
                # Turn Missions: Bias toward the inside of the turn
                if self.current_mission == "LEFT":
                    lane_center = left_x + 0.35 * lane_width
                elif self.current_mission == "RIGHT":
                    lane_center = right_x - 0.35 * lane_width
            
            self.vectors_available = True
            if self.learn_lane_width:
                # Tight bounds on learning to keep the 'center' reference stable
                if 150.0 < lane_width < (img_w * 0.65):
                    self.learned_lane_width = (
                        0.05 * lane_width + 0.95 * self.learned_lane_width)
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
            if self.current_mission == "LEFT":
                offset = 0.40 * lane_width
            elif self.current_mission == "RIGHT":
                offset = 0.60 * lane_width
            else: # STRAIGHT
                offset = 0.50 * lane_width

            if self.last_single_side == 'LEFT':
                lane_center = aim + offset
            else:
                lane_center = aim - (lane_width - offset)
            self.vectors_available = True
        else:
            self.vectors_available = False
            return

        self.last_vector_time = now
        raw_error = (lane_center - img_center) / img_center
        self.error = max(-1.0, min(1.0, raw_error))
        self.target_turn = self._compute_pid(self.error, now)
        self.target_speed = self._compute_speed(self.target_turn)
        self.last_good_turn = self.target_turn

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

    def _compute_speed(self, turn):
        severity = min(1.0, abs(turn))
        return self.speed_straight - severity * (self.speed_straight - self.speed_sharp)

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

    def control_loop(self):
        now = time.time()
        if self.obstacle_detected:
            want_turn = max(TURN_MIN, min(TURN_MAX,
                            0.35 * self.target_turn + self.obstacle_turn))
            want_speed = self.obstacle_speed
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
            self.get_logger().info(
                f"vec={'Y' if self.vectors_available else 'N'} "
                f"side={self.last_single_side} "
                f"width={self.learned_lane_width:.0f} "
                f"mission={self.current_mission} "
                f"err={self.error:+.3f} "
                f"obs={'Y' if self.obstacle_detected else 'N'}@{self.nearest_dist:.2f} "
                f"turn_int={final_turn:+.3f} "
                f"joy={self.steer_sign * final_turn:+.3f} spd={final_speed:.2f}")

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
