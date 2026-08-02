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

import re
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Bool
import cv2
import numpy as np

# Municipality Server communication message.
from synapse_msgs.msg import ServerCommunication

# --- Primary QR detector: pyzbar (more robust than cv2.QRCodeDetector) ----
# Falls back gracefully to OpenCV-only detection if pyzbar is not installed.
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    pyzbar = None
    PYZBAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# ASSUMPTIONS ABOUT synapse_msgs/msg/ServerCommunication FIELDS
# ---------------------------------------------------------------------------
# The exact field names below are assumed based on the task specification.
# If the actual .msg definition uses different field names, update ONLY
# this section (and the small number of places that reference these
# constants) to match the real message definition.
SERVER_FIELD_SRC = 'src'    # uint8 - sender id
SERVER_FIELD_DEST = 'dest'  # uint8 - recipient id
SERVER_FIELD_UID = 'uid'    # uint8 - message id, wraps 0..255
SERVER_FIELD_ACK = 'ack'    # uint8 - 0 = request/update, 1 = acknowledgement
SERVER_FIELD_MSG = 'msg'    # string - payload

# Buggy / Server IDs.
BUGGY_ID = 1
SERVER_ID = 2

# uint8 wraparound bound for the UID counter.
UID_MAX = 255

# Patient mapping: src field -> patient name and goal letter.
PATIENT_SRC_MAP = {1: "PATIENT_1", 2: "PATIENT_2", 3: "PATIENT_3"}
PATIENT_GOAL_MAP = {1: "A", 2: "B", 3: "C"}

# Hospital mapping: dest field -> hospital name and goal letter.
HOSPITAL_DEST_MAP = {4: "HOSPITAL_1", 5: "HOSPITAL_2", 6: "HOSPITAL_3"}
HOSPITAL_GOAL_MAP = {4: "X", 5: "Y", 6: "Z"}

# Default timeout (seconds) after which the same QR value may be
# re-published/re-sent even if it has not changed. Configurable.
QR_REPUBLISH_TIMEOUT_SEC = 5.0

# Toggle for verbose debug logging.
DEBUG_LOGGING = False


class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic.
    It also communicates detected QR payloads to the Municipality Server
    over `/ServerCommunication`, handling server ACKs and mission updates.

    QR detection strategy:
      1. pyzbar is tried first (more robust to rotation/noise) across a
         set of preprocessed image variants.
      2. If pyzbar finds nothing (or is unavailable), cv2.QRCodeDetector
         is tried across the same preprocessed variants.
      3. Detection stops immediately on the first success.
    """

    def __init__(self):
        super().__init__('qr_detector')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        # --- Safe Zone subscriber (unified — replaces patient_zone + hospital_zone) ---
        self.subscription_safe_zone = self.create_subscription(
            Bool,
            '/safe_zone',
            self.safe_zone_callback,
            10)

        # --- Target QR publisher (unified — notifies Line Follower of target match) ---
        self.publisher_target_qr = self.create_publisher(
            String,
            '/target_qr',
            10)

        # --- Resume Line Following publisher ----------------------------
        self.publisher_resume = self.create_publisher(
            String,
            '/resume_line_following',
            10)

        # --- Municipality Server communication ---------------------------
        # Publisher: send QR detections / ACKs to the server.
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            10)

        # Subscriber: the server replies on the same topic.
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            10)

        # --- Single, reusable QR detector instance ------------------------
        # Creating this once (instead of per-frame) avoids repeated
        # allocation overhead and improves detection reliability/speed.
        self.qr_detector = cv2.QRCodeDetector()

        # --- Reusable CLAHE object -----------------------------------------
        # Created once in __init__ (not per-frame) for performance, since
        # cv2.createCLAHE() allocates internal state each time it is called.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if not PYZBAR_AVAILABLE:
            self.get_logger().warn(
                "pyzbar not available. Falling back to cv2.QRCodeDetector only. "
                "Install with: pip install pyzbar")

        # --- Duplicate filtering state --------------------------------
        self.last_qr_value = None   # Last QR string that was published/sent.
        self.last_qr_time = None    # rclpy Time of the last publish/send.

        # --- Mission state machine ------------------------------------
        self.current_patient = "PATIENT_1"
        self.current_patient_id = 1
        self.current_goal = "A"
        self.pending_patient_qr = None
        self.stored_patient_qr = None
        self.assigned_hospital = None
        self.assigned_hospital_id = None
        self.hospital_found = False
        self.stored_hospital_qr = None
        self.waiting_for_patient_zone = False
        self.waiting_for_hospital_assignment = False
        self.waiting_for_hospital_zone = False
        self.waiting_for_new_mission = False
        self.mission_complete = False
        self.patient_zone_confirmed = False
        self.hospital_zone_confirmed = False
        self.patient_qr_published = False
        self.hospital_qr_published = False

        # --- Server communication state -----------------------------------
        self.uid_counter = 0                 # uint8 counter, starts at 0.
        self.pending_acks = {}               # uid -> qr value awaiting ACK.
        self.current_target = None           # Latest mission target stored.

        self.get_logger().info("QR Detector Node started. Waiting for images...")
        self.get_logger().info(
            "====================================\n"
            "Initial Mission\n"
            "Patient : PATIENT_1\n"
            "Goal    : A\n"
            "====================================")

    # -----------------------------------------------------------------
    # Helper: extract hospital name from QR string
    # -----------------------------------------------------------------
    @staticmethod
    def _extract_hospital_name(qr_data):
        """
        Extract the hospital identifier from a QR string like
        '{LOC: HOSPITAL_2}' or 'HOSPITAL_2'.  Returns the extracted
        name (e.g. 'HOSPITAL_2') or None if no hospital name is found.
        """
        # Try pattern: {LOC: HOSPITAL_X} or similar
        match = re.search(r'(HOSPITAL_\w+)', qr_data, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    # -----------------------------------------------------------------
    # Helper: extract patient name from QR string
    # -----------------------------------------------------------------
    @staticmethod
    def _extract_patient_name(qr_data):
        """
        Extract the patient identifier from a QR string like
        '{LOC: PATIENT_1}' or 'PATIENT_1'.  Returns the extracted
        name (e.g. 'PATIENT_1') or None if no patient name is found.
        """
        match = re.search(r'(PATIENT_\w+)', qr_data, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    # -----------------------------------------------------------------
    # Camera callback
    # -----------------------------------------------------------------
    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        # --- Convert compressed image message to OpenCV format -----------
        # Wrapped in try/except: malformed or partial frames should not
        # crash the node.
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Failed to decode compressed image: {e}")
            return

        if image is None or image.size == 0:
            self.get_logger().debug("Received empty/undecodable image frame.")
            return

        # --- Detect QR code, draw overlay ---------------------------------
        qr_data, bbox = self.detect_qr_code(image)

        if bbox is not None:
            self.draw_qr_overlay(image, bbox, qr_data)

        if qr_data:
            # Duplicate filtering: only publish/send if the QR changed, or
            # the configurable timeout since the last send has elapsed.
            self.handle_qr_detection(qr_data)

        # --- Visualization -------------------------------------------------
        try:
            cv2.imshow("QR Detector", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quit key pressed. Closing OpenCV window.")
                cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().debug(f"OpenCV visualization failed: {e}")

    # ===================================================================
    # QR DETECTION PIPELINE
    # ===================================================================
    #
    # detect_qr_code() is the single entry point used by the camera
    # callback. It orchestrates:
    #
    #   preprocess_images()   -> generate several enhanced versions of
    #                            the frame (grayscale, thresholds, CLAHE,
    #                            upscaled copies, etc.)
    #   detect_with_pyzbar()  -> try pyzbar on every preprocessed version
    #   detect_with_opencv()  -> fallback: try cv2.QRCodeDetector on every
    #                            preprocessed version
    #
    # Detection stops at the first successful decode from either method,
    # and only ONE decoded string is ever returned.
    # ===================================================================

    def preprocess_images(self, image):
        """
        Build several preprocessed variants of the input frame to maximize
        the chance of detecting QR codes that are small, blurry, rotated,
        noisy, or otherwise degraded (e.g. Gazebo-simulated textures).

        Returns:
            List of tuples: (name, processed_image, scale_back)
            where scale_back = (sx, sy) is the multiplier needed to convert
            coordinates found in `processed_image` back into the ORIGINAL
            image's coordinate space (needed for correctly drawing bboxes
            when a variant was resized).
        """
        versions = []

        # 1) Original color image - fastest path, tried first.
        versions.append(('original', image, (1.0, 1.0)))

        # Grayscale is the base for most subsequent variants.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        versions.append(('grayscale', gray, (1.0, 1.0)))

        # 2) Histogram equalization - helps with poor/uneven lighting.
        try:
            hist_eq = cv2.equalizeHist(gray)
            versions.append(('hist_eq', hist_eq, (1.0, 1.0)))
        except Exception as e:
            self.get_logger().debug(f"hist_eq preprocessing failed: {e}")

        # 3) Gaussian blur - counter-intuitively can help by smoothing
        #    sensor/compression noise before thresholding.
        try:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            versions.append(('gaussian_blur', blur, (1.0, 1.0)))
        except Exception as e:
            self.get_logger().debug(f"gaussian_blur preprocessing failed: {e}")

        # 4) Adaptive threshold - handles uneven/gradient lighting across
        #    the QR board.
        try:
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2)
            versions.append(('adaptive_threshold', adaptive, (1.0, 1.0)))
        except Exception as e:
            self.get_logger().debug(f"adaptive_threshold preprocessing failed: {e}")

        # 5) Otsu threshold - good global binarization for evenly-lit
        #    simulated (Gazebo) QR codes.
        try:
            _, otsu = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            versions.append(('otsu', otsu, (1.0, 1.0)))
        except Exception as e:
            self.get_logger().debug(f"otsu preprocessing failed: {e}")

        # 6) CLAHE (contrast-limited adaptive histogram equalization) -
        #    boosts local contrast, useful for noisy/low-contrast frames.
        try:
            clahe_img = self.clahe.apply(gray)
            versions.append(('clahe', clahe_img, (1.0, 1.0)))
        except Exception as e:
            self.get_logger().debug(f"clahe preprocessing failed: {e}")

        # 7) 2x resize - helps detect small/distant QR codes by giving the
        #    detector more pixels to work with.
        try:
            resized_2x = cv2.resize(
                gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            # Coordinates found in this 2x image must be scaled by 0.5 to
            # map back to the original image.
            versions.append(('resize_2x', resized_2x, (0.5, 0.5)))
        except Exception as e:
            self.get_logger().debug(f"resize_2x preprocessing failed: {e}")

        # 8) 3x resize - even more aggressive upscaling for very small QR
        #    codes, at extra compute cost (tried last).
        try:
            resized_3x = cv2.resize(
                gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            versions.append(('resize_3x', resized_3x, (1.0 / 3.0, 1.0 / 3.0)))
        except Exception as e:
            self.get_logger().debug(f"resize_3x preprocessing failed: {e}")

        return versions

    def detect_with_pyzbar(self, versions):
        """
        Attempt QR decoding with pyzbar across all preprocessed variants.
        Stops and returns immediately on the first successful decode.

        Returns:
            (data, points) where points is an (N, 2) float32 array of
            corner coordinates in ORIGINAL image space, or (None, None)
            if nothing was decoded / pyzbar is unavailable.
        """
        if not PYZBAR_AVAILABLE:
            return None, None

        for name, img, scale in versions:
            try:
                decoded_objects = pyzbar.decode(img)
            except Exception as e:
                self.get_logger().debug(f"pyzbar failed on '{name}': {e}")
                continue

            for obj in decoded_objects:
                try:
                    data = obj.data.decode('utf-8').strip()
                except Exception as e:
                    self.get_logger().debug(f"pyzbar payload decode failed: {e}")
                    continue

                if not data:
                    continue

                # Prefer the polygon (4+ pts) since it follows rotation
                # more accurately than the axis-aligned rect.
                if obj.polygon and len(obj.polygon) >= 4:
                    points = np.array(
                        [[p.x, p.y] for p in obj.polygon], dtype=np.float32)
                else:
                    x, y, w, h = obj.rect
                    points = np.array(
                        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        dtype=np.float32)

                # Scale coordinates back to the original image's space.
                sx, sy = scale
                points[:, 0] *= sx
                points[:, 1] *= sy

                if DEBUG_LOGGING:
                    self.get_logger().debug(
                        f"pyzbar detected on '{name}': {data}")

                return data, points

        return None, None

    def detect_with_opencv(self, versions):
        """
        Fallback QR decoding using cv2.QRCodeDetector across all
        preprocessed variants. Stops and returns immediately on the first
        successful decode.

        Returns:
            (data, points) where points is an (N, 2) float32 array of
            corner coordinates in ORIGINAL image space, or (None, None)
            if nothing was decoded.
        """
        for name, img, scale in versions:
            try:
                data, bbox, _ = self.qr_detector.detectAndDecode(img)
            except cv2.error as e:
                self.get_logger().debug(f"OpenCV QR detection failed on '{name}': {e}")
                continue
            except Exception as e:
                self.get_logger().debug(
                    f"Unexpected OpenCV QR detection error on '{name}': {e}")
                continue

            if bbox is not None and data:
                data = data.strip()
                if not data:
                    continue

                points = bbox.reshape(-1, 2).astype(np.float32)
                sx, sy = scale
                points[:, 0] *= sx
                points[:, 1] *= sy

                if DEBUG_LOGGING:
                    self.get_logger().debug(
                        f"OpenCV detected on '{name}': {data}")

                return data, points

        return None, None

    def detect_qr_code(self, image):
        """
        Main QR detection entry point.

        Order of operations (stops at first success):
          1. Build preprocessed image variants (once, shared by both
             detectors to avoid redundant work).
          2. Try pyzbar on every variant.
          3. If pyzbar found nothing, try cv2.QRCodeDetector on every
             variant.

        Returns:
            (data, points) - decoded string and bbox corner points in
            ORIGINAL image coordinates, or (None, None) if no QR found.
        """
        versions = self.preprocess_images(image)

        # --- Primary: pyzbar ------------------------------------------------
        data, points = self.detect_with_pyzbar(versions)
        if data:
            return data, points

        # --- Fallback: OpenCV ------------------------------------------------
        data, points = self.detect_with_opencv(versions)
        if data:
            return data, points

        return None, None

    def draw_qr_overlay(self, image, bbox, qr_data):
        """
        Draws the QR bounding box and decoded text onto the frame.

        Works for both pyzbar-derived and OpenCV-derived corner points,
        since both are normalized to an (N, 2) array in `detect_qr_code()`.
        """
        try:
            pts = bbox.reshape(-1, 2).astype(int)
            for i in range(len(pts)):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % len(pts)])
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            if qr_data:
                text_origin = (int(pts[0][0]), max(int(pts[0][1]) - 10, 0))
                cv2.putText(
                    image, qr_data, text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            self.get_logger().debug(f"Failed to draw QR overlay: {e}")

    # -----------------------------------------------------------------
    # Duplicate filtering + publishing + server notification
    # -----------------------------------------------------------------
    def handle_qr_detection(self, qr_data):
        """
        Decides whether this QR detection is "new enough" to publish and
        route appropriately.

        The QR is always published on /qr_detection.

        Routing logic:
          - If waiting for new mission: skip routing (only raw publish).
          - If the QR is a Hospital QR and we have an assigned hospital:
              - Extract the hospital name from the QR string.
              - If it matches the assigned hospital, publish on /target_qr
                and wait for /safe_zone.
              - If it does not match, ignore it (wrong hospital).
          - If the QR is a Patient QR:
              - Extract the patient name from the QR string.
              - If it matches the current_patient, store, publish on /target_qr,
                and wait for /safe_zone.
              - If it does not match, ignore it (wrong patient).
          - Otherwise: ignore (unknown QR type).
        """
        now = self.get_clock().now()
        should_send = False

        if qr_data != self.last_qr_value:
            should_send = True
        elif self.last_qr_time is not None:
            elapsed_sec = (now - self.last_qr_time).nanoseconds / 1e9
            if elapsed_sec >= QR_REPUBLISH_TIMEOUT_SEC:
                should_send = True
        else:
            should_send = True

        if not should_send:
            return

        self.last_qr_value = qr_data
        self.last_qr_time = now

        # Always publish the decoded QR payload on /qr_detection.
        msg = String()
        msg.data = qr_data
        self.publisher_qr.publish(msg)
        self.get_logger().info(f"Published QR Data: {qr_data}")

        # Don't route QR codes while waiting for new mission from server.
        if self.waiting_for_new_mission:
            return

        # ---- Hospital QR handling ----
        if self.assigned_hospital is not None:
            detected_hospital = self._extract_hospital_name(qr_data)
            if detected_hospital is not None:
                if detected_hospital == self.assigned_hospital.upper():
                    # Correct hospital found — publish on /target_qr.
                    self.hospital_found = True
                    self.stored_hospital_qr = qr_data
                    self.waiting_for_hospital_zone = True
                    self.hospital_qr_published = True
                    target_qr_msg = String()
                    target_qr_msg.data = qr_data
                    self.publisher_target_qr.publish(target_qr_msg)
                    self.get_logger().info(
                        "====================================\n"
                        "Correct Hospital Found\n"
                        f"Detected : {detected_hospital}\n"
                        f"Assigned : {self.assigned_hospital}\n"
                        "Publishing /target_qr\n"
                        "Waiting for Safe Zone...\n"
                        "====================================")
                else:
                    # Wrong hospital — ignore completely.
                    self.get_logger().info(
                        "====================================\n"
                        "Wrong Hospital\n"
                        f"Detected : {detected_hospital}\n"
                        f"Assigned : {self.assigned_hospital}\n"
                        "Continuing Search...\n"
                        "====================================")
                return

        # ---- Patient QR handling ----
        detected_patient = self._extract_patient_name(qr_data)
        if detected_patient is not None:
            if detected_patient == self.current_patient.upper():
                # Correct patient — store and publish /target_qr.
                self.pending_patient_qr = qr_data
                self.stored_patient_qr = qr_data
                self.waiting_for_patient_zone = True
                self.patient_qr_published = True
                target_qr_msg = String()
                target_qr_msg.data = qr_data
                self.publisher_target_qr.publish(target_qr_msg)
                self.get_logger().info(
                    "====================================\n"
                    "Correct Patient Found\n"
                    f"Detected : {detected_patient}\n"
                    f"Assigned : {self.current_patient}\n"
                    "Publishing /target_qr\n"
                    "Waiting for Safe Zone...\n"
                    "====================================")
            else:
                # Wrong patient — ignore completely.
                self.get_logger().info(
                    "====================================\n"
                    "Wrong Patient\n"
                    f"Detected : {detected_patient}\n"
                    f"Assigned : {self.current_patient}\n"
                    "Continuing Search...\n"
                    "====================================")
            return

    # -----------------------------------------------------------------
    # Safe Zone callback (unified — replaces patient_zone + hospital_zone)
    # -----------------------------------------------------------------
    def safe_zone_callback(self, message):
        """
        Handles incoming /safe_zone messages. When the safe zone is
        confirmed (True), routes the stored QR to the Municipality Server
        based on the current mission phase.

        - If waiting_for_patient_zone: sends patient QR and transitions
          to waiting_for_hospital_assignment.
        - If waiting_for_hospital_zone: sends hospital QR and signals
          the Line Follower that the mission is complete.
        """
        if not message.data:
            return

        # --- Patient safe zone confirmed ---
        if self.waiting_for_patient_zone and self.pending_patient_qr is not None:
            self.get_logger().info(
                "====================================\n"
                "Safe Zone Confirmed (Patient)\n"
                "Sending Patient QR\n"
                "Waiting For Hospital Assignment...\n"
                "====================================")
            self.send_qr_to_server(self.pending_patient_qr)
            self.pending_patient_qr = None
            self.waiting_for_patient_zone = False
            self.waiting_for_hospital_assignment = True
            self.patient_zone_confirmed = True
            return

        # --- Hospital safe zone confirmed ---
        if self.waiting_for_hospital_zone and self.stored_hospital_qr is not None:
            self.get_logger().info(
                "====================================\n"
                "Safe Zone Confirmed (Hospital)\n"
                "Sending Hospital QR\n"
                "====================================")
            self.send_qr_to_server(self.stored_hospital_qr)
            self.hospital_zone_confirmed = True

            # Do NOT call _reset_mission_state() here.
            # Do NOT erase mission variables.
            self.mission_complete = True
            self.waiting_for_new_mission = True

            # Signal the Line Follower that the mission is complete.
            # The Line Follower will transition from WAITING_FOR_SERVER_ACK
            # to MISSION_COMPLETE.
            complete_msg = String()
            complete_msg.data = "MISSION_COMPLETE"
            self.publisher_resume.publish(complete_msg)
            self.get_logger().info(
                "Published /resume_line_following MISSION_COMPLETE")
            return

    # -----------------------------------------------------------------
    # Mission state reset for new patient
    # -----------------------------------------------------------------
    def _reset_mission_state_for_new_patient(self):
        """Reset only mission-related state for a new patient mission.
        Does NOT reset current_patient, current_patient_id, or current_goal
        (they are updated after this call). Does NOT reset ROS
        publishers/subscribers. Does NOT recreate the node."""
        self.pending_patient_qr = None
        self.stored_patient_qr = None
        self.assigned_hospital = None
        self.assigned_hospital_id = None
        self.stored_hospital_qr = None
        self.hospital_found = False
        self.waiting_for_patient_zone = False
        self.waiting_for_hospital_assignment = False
        self.waiting_for_hospital_zone = False
        self.mission_complete = False
        self.waiting_for_new_mission = False
        self.patient_zone_confirmed = False
        self.hospital_zone_confirmed = False
        self.patient_qr_published = False
        self.hospital_qr_published = False
        # Duplicate QR cache reset.
        self.last_qr_value = None
        self.last_qr_time = None

    # -----------------------------------------------------------------
    # Municipality Server communication
    # -----------------------------------------------------------------
    def get_next_uid(self):
        """Returns the current UID and increments the counter with
        uint8 wraparound (255 -> 0)."""
        uid = self.uid_counter
        self.uid_counter = (self.uid_counter + 1) % (UID_MAX + 1)
        return uid

    def send_qr_to_server(self, qr_data):
        """Sends a freshly detected QR value to the server as a new
        request (ack = 0), tracking the UID for pending ACK matching."""
        uid = self.get_next_uid()

        server_msg = ServerCommunication()
        setattr(server_msg, SERVER_FIELD_SRC, BUGGY_ID)
        setattr(server_msg, SERVER_FIELD_DEST, SERVER_ID)
        setattr(server_msg, SERVER_FIELD_UID, uid)
        setattr(server_msg, SERVER_FIELD_ACK, 0)
        setattr(server_msg, SERVER_FIELD_MSG, qr_data)

        # Track this UID until the server acknowledges it.
        self.pending_acks[uid] = qr_data

        self.publisher_server.publish(server_msg)
        self.get_logger().info("Sent to server")

        if DEBUG_LOGGING:
            self.get_logger().debug(f"Server request: uid={uid}, msg={qr_data}")

    def server_communication_callback(self, message):
        """Handles all incoming /ServerCommunication messages.

        State-aware processing order:
          1. Ignore self-published packets.
          2. Hospital assignment (only when waiting_for_hospital_assignment).
          3. New patient assignment (only when waiting_for_new_mission).
          4. Generic ACK (only when uid matches a pending QR request).
          5. INVALID notification.
          6. OK notification.
          7. Generic mission update.
        """
        dest = getattr(message, SERVER_FIELD_DEST)
        src = getattr(message, SERVER_FIELD_SRC)
        ack = getattr(message, SERVER_FIELD_ACK)
        uid = getattr(message, SERVER_FIELD_UID)
        payload = getattr(message, SERVER_FIELD_MSG)

        # ============================================================
        # 1. Ignore self-published packets (both requests and ACKs).
        # ============================================================
        if src == BUGGY_ID and dest == SERVER_ID:
            self.get_logger().debug(
                f"Ignored self-published packet uid={uid}"
            )
            return

        # ============================================================
        # 2. Hospital assignment
        #    Valid ONLY when waiting_for_hospital_assignment == True.
        #    Server packet: src = completed patient ID, dest = assigned hospital.
        #    Validate: src must match current_patient_id.
        # ============================================================
        if (
            self.waiting_for_hospital_assignment
            and dest in HOSPITAL_DEST_MAP
        ):
            # Validate that the completed patient matches our current patient.
            if src != self.current_patient_id:
                self.get_logger().info(
                    "Ignoring hospital assignment: patient mismatch\n"
                    f"Packet src (completed patient) : {src}\n"
                    f"Current patient ID             : {self.current_patient_id}")
                return

            self.assigned_hospital = HOSPITAL_DEST_MAP[dest]
            self.assigned_hospital_id = dest
            hospital_goal = HOSPITAL_GOAL_MAP[dest]
            self.current_goal = hospital_goal
            self.waiting_for_hospital_assignment = False

            self.get_logger().info(
                "====================================\n"
                "Hospital Assignment Received\n"
                f"Hospital : {self.assigned_hospital}\n"
                f"Goal      : {hospital_goal}\n"
                "Publishing Resume\n"
                "====================================")

            # Publish /resume_line_following
            resume_msg = String()
            resume_msg.data = "RESUME"
            self.publisher_resume.publish(resume_msg)

            # Send mandatory ACK back to server
            ack_msg = ServerCommunication()
            setattr(ack_msg, SERVER_FIELD_SRC, BUGGY_ID)
            setattr(ack_msg, SERVER_FIELD_DEST, SERVER_ID)
            setattr(ack_msg, SERVER_FIELD_UID, uid)
            setattr(ack_msg, SERVER_FIELD_ACK, 1)
            setattr(ack_msg, SERVER_FIELD_MSG, "")
            self.publisher_server.publish(ack_msg)

            return

        # ============================================================
        # 3. Next patient assignment
        #    Valid ONLY when waiting_for_new_mission == True.
        #    Server packet: src = completed hospital ID, dest = next patient ID.
        #    Validate: src must match assigned_hospital_id.
        #    Use dest to determine the next patient and goal.
        #    MUST be checked BEFORE the generic ACK handler.
        # ============================================================
        if (
            self.waiting_for_new_mission
            and dest in PATIENT_SRC_MAP
            and ack == 1
        ):
            # Validate that the completed hospital matches our assigned hospital.
            if self.assigned_hospital_id is not None and src != self.assigned_hospital_id:
                self.get_logger().info(
                    "Ignoring next patient assignment: hospital mismatch\n"
                    f"Packet src (completed hospital) : {src}\n"
                    f"Assigned hospital ID             : {self.assigned_hospital_id}")
                return

            new_patient = PATIENT_SRC_MAP[dest]
            new_goal = PATIENT_GOAL_MAP[dest]

            # Reset mission state for new patient.
            self._reset_mission_state_for_new_patient()

            # Update current patient and goal.
            self.current_patient = new_patient
            self.current_patient_id = dest
            self.current_goal = new_goal

            self.get_logger().info(
                "====================================\n"
                "NEW PATIENT MISSION\n"
                f"Patient : {new_patient}\n"
                f"Goal    : {new_goal}\n"
                "Mission Reset Complete\n"
                "====================================")

            # Publish /resume_line_following
            resume_msg = String()
            resume_msg.data = "RESUME"
            self.publisher_resume.publish(resume_msg)

            # Send ACK back to server.
            ack_msg = ServerCommunication()
            setattr(ack_msg, SERVER_FIELD_SRC, BUGGY_ID)
            setattr(ack_msg, SERVER_FIELD_DEST, SERVER_ID)
            setattr(ack_msg, SERVER_FIELD_UID, uid)
            setattr(ack_msg, SERVER_FIELD_ACK, 1)
            setattr(ack_msg, SERVER_FIELD_MSG, "")
            self.publisher_server.publish(ack_msg)

            return

        # ============================================================
        # 4. Generic ACK processing (ack == 1)
        #    Only consume the packet when the ACK actually matches a
        #    pending QR request. If ack==1 but uid is NOT in
        #    pending_acks, do NOT return — continue processing so
        #    the packet can still be handled by INVALID/OK/generic
        #    blocks below.
        # ============================================================
        if ack == 1:
            if uid in self.pending_acks:
                sent_value = self.pending_acks.pop(uid)
                if DEBUG_LOGGING:
                    self.get_logger().debug(
                        f"ACK matches QR '{sent_value}' (uid={uid})")
                return
            else:
                self.get_logger().info(
                    "ACK received\n"
                    "UID not found in pending list\n"
                    "Continuing callback...")

        # From here, ack == 0 or unmatched ack==1: server-initiated request/update.

        # ============================================================
        # 5. INVALID notification
        # ============================================================
        if payload == "INVALID":
            self.get_logger().info("INVALID received")
            return

        # ============================================================
        # 6. OK notification
        # ============================================================
        if payload == "OK":
            self.get_logger().info("Parking successful")
            return

        # ============================================================
        # 7. Generic mission update
        # ============================================================
        self.get_logger().info(f"Mission update:\n{payload}")
        self.current_target = payload

        # Mandatory acknowledgement back to server, echoing the same uid.
        ack_msg = ServerCommunication()
        setattr(ack_msg, SERVER_FIELD_SRC, BUGGY_ID)
        setattr(ack_msg, SERVER_FIELD_DEST, SERVER_ID)
        setattr(ack_msg, SERVER_FIELD_UID, uid)
        setattr(ack_msg, SERVER_FIELD_ACK, 1)
        setattr(ack_msg, SERVER_FIELD_MSG, "")
        self.publisher_server.publish(ack_msg)

    # -----------------------------------------------------------------
    # Clean shutdown
    # -----------------------------------------------------------------
    def destroy_node(self):
        """Ensures OpenCV windows are destroyed on node shutdown."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
