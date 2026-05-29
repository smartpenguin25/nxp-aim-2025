# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
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
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):

		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		



		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)


	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		if row < self.row_count and col < self.col_count:
			self.boxes[row][col].delete(1.0, tk.END)
			self.boxes[row][col].insert(tk.END, text)
		else:
			print(f"[WARNING] GUI index out of range: row={row}, col={col}")

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')

		self.exploration_phase = 'exploring'  # 'exploring', 'shelf_inspection', 'completed'
		self.exploration_completed = False
		self.consecutive_no_frontiers = 0
		self.max_no_frontiers = 3  # Stop exploration after 3 consecutive failed attempts
		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0
		self.max_no_frontiers = 3
		self.table_row_count = 0
		self.table_col_count = 0
		self.scan_timer = None

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()

		# --- Shelf Detection ---
		self.detected_shelves = []  # List to store detected shelf locations
		self.shelf_dimensions = (1.35, 0.55)  # Shelf dimensions in meters
		self.visited_shelves = set()  # Track visited shelves
		self.current_shelf_id = 1
		self.next_shelf_angle = self.initial_angle  # From parameter

		# --- QR Code Processing ---
		self.qr_detection_enabled = True
		self.last_qr_decode_time = time.time()
		self.qr_decode_cooldown = 2.0  # seconds


        # **NEW**: Goal queue system for shelf inspection
		self.goal_queue = []
		self.current_inspection_shelf = None
		self.inspection_stage = 'idle'  # 'idle', 'qr_scan', 'object_scan'
		
		# **NEW**: Recovery logic
		self.recovery_mode = False
		self.recovery_attempts = 0
		self.max_recovery_attempts = 3

		self.shelves_detected = False
		self.last_shelf_detection_time = 0
		self.shelf_detection_cooldown = 5.0  # seconds
		
		# **NEW**: Add QR scanning state
		self.qr_scanning_active = False
		self.qr_scan_start_time = 0
		self.qr_scan_duration = 5.0  # seconds to scan for QR
		
		# **NEW**: Add object scanning state  
		self.object_scanning_active = False
		self.object_scan_start_time = 0
		self.object_scan_duration = 5.0 # seconds to scan for objects

        # **NEW**: Scanning state management
		self.scanning_in_progress = False
		self.scanning_start_time = 0
		self.scanning_duration = 10.0 # seconds to scan at each position
		self.current_scan_type = None  # 'qr' or 'object'
		self.qr_found_at_position = False
		self.scan_delay_timer = None # <--- ADD THIS LINE
		#abhi
		#self.spawn_pose = None  # To store the initial pose
		#self.spawn_pose_set = False
		self.shelf_id_mapping = {}  # Maps shelf_id to shelf object
		self.unidentified_shelves = []  # Shelves we've detected but haven't identified yet
		self.current_target_shelf_id = 1  # Start by looking for shelf 1
		self.successfully_scanned_shelves = set()  # Track which shelf IDs we've successfully scanned
		self.accumulated_objects = {}
		self.last_confirmed_shelf_center = None
		self.object_confidence_threshold = 2  # Exit early after detecting objects twice
		self.object_detection_count = 0
		self.frame_captures = []  # Store object detections for each frame
		self.side_a_frames = []  # Store frames for side A
		self.side_b_frames = []  # Store frames for side B
		self.current_side = 'A'  # Track which side we're scanning
		self.best_frame_side_a = None
		self.best_frame_side_b = None
		self.frame_capture_timer = None
		self.frame_capture_count = 0
		self.frames_per_position = 3  # Capture 3 frames at each position
		self.frame_capture_interval = 1.0  # 1 second between frames
		self.max_objects_per_shelf = 6  # Maximum objects constraint

	def finish_exploration(self):
		"""Mark exploration as completed and start shelf inspection."""
		self.exploration_phase = 'shelf_inspection'
		self.exploration_completed = True
		self.logger.info("=== EXPLORATION COMPLETED - Starting shelf inspection ===")
		
		# Reset exploration parameters
		self.consecutive_no_frontiers = 0
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		# Make sure this is initialized
		self.last_confirmed_shelf_center = None
	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)
		#abhi
		# if not self.spawn_pose_set:
		# 	self.spawn_pose = (self.buggy_pose_x, self.buggy_pose_y)
		# 	self.spawn_pose_set = True

	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates and detect shelves."""
		self.simple_map_curr = message

		# Always detect shelves from map
		self.detect_shelves_from_map(message)

		# Guard condition: Do not make high-level decisions if the robot is busy.
		if (self.exploration_phase != 'shelf_inspection' or 
			not self.goal_completed or 
			self.scanning_in_progress):
			return
		
		# --- From here on, we are idle and in the inspection phase ---

		# Main Decision Logic:
		if not self.current_inspection_shelf:
			# Check if we need to move to next target
			if self.current_target_shelf_id - 1 in self.successfully_scanned_shelves:
				# We successfully scanned the previous target, look for next
				self.logger.info(f"Decision: Looking for shelf {self.current_target_shelf_id}")
				self.navigate_to_next_shelf()
			elif self.current_target_shelf_id == 1:
				# First shelf
				self.logger.info(f"Decision: Looking for initial shelf (shelf 1)")
				self.navigate_to_next_shelf()
			else:
				# Previous target not found yet, keep looking
				self.logger.info(f"Decision: Still looking for shelf {self.current_target_shelf_id}")
				self.navigate_to_next_shelf()
				
	def global_map_callback(self, message):
		"""Callback function to handle global map updates."""
		self.global_map_curr = message

		# **NEW**: Only explore if in exploration phase
		if self.exploration_phase != 'exploring' or not self.goal_completed:
			return
		
		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		frontiers = self.get_frontiers_for_space_exploration(map_array)

		map_info = self.global_map_curr.info
		if frontiers:
			# **NEW**: Reset counter when frontiers found
			self.consecutive_no_frontiers = 0
			
			closest_frontier = None
			min_distance_curr = float('inf')

			for fy, fx in frontiers:
				fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy, map_info)
				distance = euclidean((fx_world, fy_world), self.buggy_center)
				if (distance < min_distance_curr and
					distance <= self.max_step_dist_world_meters and
					distance >= self.min_step_dist_world_meters):
					min_distance_curr = distance
					closest_frontier = (fy, fx)

			if closest_frontier:
				fy, fx = closest_frontier
				goal = self.create_goal_from_map_coord(fx, fy, map_info)
				self.send_goal_from_world_pose(goal)
				print("Sending goal for space exploration.")
				return
			else:
				self.max_step_dist_world_meters += 2.0
				new_min_step_dist = self.min_step_dist_world_meters - 1.0
				self.min_step_dist_world_meters = max(0.25, new_min_step_dist)
		else:
			# **NEW**: Increment counter when no frontiers found
			self.consecutive_no_frontiers += 1
			print(f"No frontiers found; consecutive count = {self.consecutive_no_frontiers}")
			
			# **NEW**: Check if exploration should end
			if self.consecutive_no_frontiers >= self.max_no_frontiers:
				self.finish_exploration()

	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers



	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
		"""Callback function to handle incoming camera images."""
		# **FIXED**: Only process QR codes during active QR scanning
		if not self.scanning_in_progress or self.current_scan_type != 'qr':
			return
		
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		
		# Process QR code detection
		if image is not None:
			qr_decoder = cv2.QRCodeDetector()
			qr_data, points, _ = qr_decoder.detectAndDecode(image)
		
		if qr_data and qr_data != "Empty":
			self.qr_code_str = qr_data
			self.qr_found_at_position = True
			self.logger.info(f"QR code found: {qr_data}")
			
			# Process QR code data
			self.process_qr_code(qr_data)
			
			# Stop scanning early since we found the QR
			self.finish_scanning()
		
		# Optional: publish debug image
		self.publish_debug_image(self.publisher_qr_decode, image)

	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def shelf_objects_callback(self, message):
		"""Enhanced callback that captures frames per side."""
		self.shelf_objects_curr = message
		
		if self.current_scan_type == 'object' and self.scanning_in_progress:
			# Only process if we get objects
			if message.object_name:
				# Calculate total objects (capped at max)
				total_objects = sum(message.object_count)
				if total_objects > self.max_objects_per_shelf:
					self.logger.warn(f"Detected {total_objects} objects, capping at {self.max_objects_per_shelf}")
					total_objects = self.max_objects_per_shelf
				
				# Store this frame's detection
				frame_data = {
					'object_name': list(message.object_name),
					'object_count': list(message.object_count),
					'total_objects': min(total_objects, self.max_objects_per_shelf)
				}
				
				# Store in appropriate side list
				if self.current_side == 'A':
					self.side_a_frames.append(frame_data)
					self.logger.info(f"Side A - Frame {len(self.side_a_frames)}: Detected {frame_data['total_objects']} objects")
				else:
					self.side_b_frames.append(frame_data)
					self.logger.info(f"Side B - Frame {len(self.side_b_frames)}: Detected {frame_data['total_objects']} objects")

	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)



	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

	def goal_result_callback(self, future):
			"""Callback function executed when the navigation goal reaches a final result."""
			status = future.result().status
			
			if status == GoalStatus.STATUS_SUCCEEDED:
				self.logger.info(f"Goal reached successfully! (Phase: {self.exploration_phase})")
				self.recovery_attempts = 0
				
				# CRITICAL FIX: The navigation action is complete. Reset the state to True
				# so that the system is ready to send a new goal later (e.g., after scanning).
				self.goal_completed = True
				
				if self.exploration_phase == 'shelf_inspection':
					# Pop the successful goal from queue
					if self.goal_queue:
						completed_goal = self.goal_queue.pop(0)
						self.logger.info(f"Successfully completed {completed_goal['type']} goal")
					
					# Now, start the scanning process at the reached position.
					'''if self.inspection_stage in ['qr_scan', 'object_scan']:
						self.logger.info(f"Starting {self.inspection_stage} at reached position")
						self.start_scanning_at_position()'''
										# ADD THIS NEW BLOCK IN ITS PLACE
					if self.inspection_stage == 'object_scan':
						# For OBJECT scanning, wait for 2 seconds before starting.
						self.logger.info("ROBOT STOPPED. Waiting 2 seconds before starting object scan...")
						self.scan_delay_timer = self.create_timer(4.0, self._start_scan_after_delay)

					elif self.inspection_stage == 'qr_scan':
						# For QR scanning, start immediately.
						self.logger.info("ROBOT STOPPED. Starting QR scan immediately.")
						self.start_scanning_at_position()
				
			else:
				self.logger.warn(f"Goal failed with status: {status}")
				if self.exploration_phase == 'shelf_inspection':
					self.handle_navigation_failure()
				else:
					# For exploration failures, just mark as completed and continue
					self.goal_completed = True

			# Clear the handle regardless of outcome.
			self.goal_handle_curr = None

	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose):
		"""Sends a navigation goal to the Nav2 action server."""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		# **NEW**: Log which phase is sending the goal
		self.logger.info(f"Sending goal in {self.exploration_phase} phase")
		
		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)

	def detect_shelves_from_map(self, map_message):
		"""
		Detect shelves from the occupancy grid map by identifying rectangular obstacles
		of the expected shelf dimensions (1.35m × 0.55m).
		"""
		map_info = map_message.info
		height, width = map_info.height, map_info.width
		resolution = map_info.resolution
		
		# Convert map data to numpy array
		map_array = np.array(map_message.data).reshape((height, width))
		
		# Create obstacle mask (cells with value 100 are obstacles)
		obstacle_mask = (map_array == 100)
		
		# Label connected components (clusters)
		labeled_array, num_features = label(obstacle_mask)
		
		potential_shelves = []
		shelf_w, shelf_h = self.shelf_dimensions  # 1.35m × 0.55m
		tolerance = 0.3  # 30cm tolerance for shelf dimensions
		
		self.logger.info(f"Found {num_features} obstacle clusters")
		
		for i in range(1, num_features + 1):
			# Get all pixels belonging to this cluster
			cluster_mask = (labeled_array == i)
			cluster_coords = np.where(cluster_mask)
			
			# Skip very small clusters (noise)
			if len(cluster_coords[0]) < 10:
				continue
			
			# Convert to (x, y) coordinates for OpenCV
			cluster_points = np.column_stack((cluster_coords[1], cluster_coords[0])).astype(np.float32)
			
			# Need at least 5 points to fit a rectangle
			if len(cluster_points) < 5:
				continue
			
			# Fit minimum area rectangle
			rect = cv2.minAreaRect(cluster_points)
			(cx, cy), (w, h), angle = rect
			
			# Convert dimensions from map pixels to world meters
			w_meters = w * resolution
			h_meters = h * resolution
			
			# Check if dimensions match shelf size (with tolerance)
			# The shelf can be oriented in either direction
			matches_shelf = (
				(abs(w_meters - shelf_w) < tolerance and abs(h_meters - shelf_h) < tolerance) or
				(abs(w_meters - shelf_h) < tolerance and abs(h_meters - shelf_w) < tolerance)
			)
			
			if matches_shelf:
				# Convert center coordinates from map to world
				world_x, world_y = self.get_world_coord_from_map_coord(cx, cy, map_info)
				
				# Convert angle to radians
				orientation = np.deg2rad(angle)
				# **FIXED**: Normalize dimensions to always be (1.35, 0.55)
			# This ensures consistent behavior regardless of detection orientation
				if w_meters > h_meters:
					normalized_dimensions = (w_meters, h_meters)
					normalized_orientation = orientation
				else:
					normalized_dimensions = (h_meters, w_meters)
					normalized_orientation = orientation + math.pi/2
					# Normalize orientation to [-π, π]
				while normalized_orientation > math.pi:
					normalized_orientation -= 2 * math.pi
				while normalized_orientation < -math.pi:
					normalized_orientation += 2 * math.pi
				
				shelf_info = {
					'center': (world_x, world_y),
					'orientation': normalized_orientation,
					'dimensions': normalized_dimensions, #always 1.35X0.55
					'visited': False,
					'pixel_count': len(cluster_coords[0])
				}
				
				potential_shelves.append(shelf_info)
				self.logger.info(f"Potential shelf found: center=({world_x:.2f}, {world_y:.2f}), "
							f"dimensions=({normalized_dimensions[0]:.2f}×{normalized_dimensions[1]:.2f}), "
							f"angle={math.degrees(normalized_orientation):.1f}°")
		
		# Merge nearby clusters (same shelf detected as multiple clusters)
		merged_shelves = []
		merge_distance = 1.0  # meters - shelves closer than this are considered the same
		
		for shelf in potential_shelves:
			merged = False
			
			for existing_shelf in merged_shelves:
				dist = euclidean(shelf['center'], existing_shelf['center'])
				
				if dist < merge_distance:
					# Merge with existing shelf - keep the one with more pixels (more detailed)
					if shelf['pixel_count'] > existing_shelf['pixel_count']:
						existing_shelf.update(shelf)
					merged = True
					break
			
			if not merged:
				merged_shelves.append(shelf)
		
		# Update detected shelves
		old_count = len(self.detected_shelves)
		self.detected_shelves = merged_shelves
		new_count = len(self.detected_shelves)
		
		self.logger.info(f"Shelf detection complete: {old_count} → {new_count} shelves")
		
		# Log all detected shelves
		for i, shelf in enumerate(self.detected_shelves):
			center = shelf['center']
			dims = shelf['dimensions']
			self.logger.info(f"Shelf {i+1}: center=({center[0]:.2f}, {center[1]:.2f}), "
							f"size=({dims[0]:.2f}×{dims[1]:.2f})")
	def process_qr_code(self, qr_data):
		"""Process decoded QR code data."""
		try:
			# Parse QR code: format "shelf_id_angle_secret"
			parts = qr_data.split('_')
			if len(parts) >= 3:
				shelf_id = int(parts[0])
				angle = float(parts[1])
				secret = '_'.join(parts[2:])  # In case secret contains underscores
				
				self.logger.info(f"QR Code decoded: Shelf {shelf_id}, Angle {angle}°, Secret: {secret}")
				
				# Store the QR data temporarily
				self.qr_code_str = qr_data
				
				# Check if this is the shelf we were looking for
				if shelf_id == self.current_target_shelf_id:
					# This is the correct shelf!
					self.logger.info(f"✓ Found correct shelf {shelf_id}!")
					
					# Update next shelf angle
					self.next_shelf_angle = angle
					
					# Mark this shelf as successfully scanned
					self.successfully_scanned_shelves.add(shelf_id)
					
					# Update shelf objects with QR
					self.shelf_objects_curr.qr_decoded = qr_data
					
					# CRITICAL FIX: Store the current shelf center NOW before incrementing target
					if self.current_inspection_shelf:
						self.last_confirmed_shelf_center = self.current_inspection_shelf['center']
						self.logger.info(f"Stored shelf {shelf_id} center immediately: {self.last_confirmed_shelf_center}")
					
					# Move to next shelf in sequence
					self.current_target_shelf_id = shelf_id + 1
					
					self.logger.info(f"Next target will be shelf {self.current_target_shelf_id} at angle {angle}°")
				else:
					# Wrong shelf!
					self.logger.warn(f"✗ Found shelf {shelf_id} but we're looking for shelf {self.current_target_shelf_id}")
					self.logger.warn(f"Ignoring this shelf and continuing search")
					# Reset QR string since it's not the shelf we want
					self.qr_code_str = "Empty"
					
		except (ValueError, IndexError) as e:
			self.logger.error(f"Error parsing QR code: {e}")


	def navigate_to_next_shelf(self):
		"""Navigate to the next shelf in sequential order using angle heuristics."""
		if not self.detected_shelves:
			self.logger.warn("No shelves detected yet")
			return False
		
		# Check if we've visited all required shelves
		if self.current_target_shelf_id > self.shelf_count:
			self.logger.info(f"✓ All {self.shelf_count} shelves have been successfully scanned!")
			return False
		
		# Get unvisited physical shelves
		unvisited_shelves = [shelf for shelf in self.detected_shelves 
							if not shelf.get('visited', False)]
		
		if not unvisited_shelves:
			self.logger.warn("All detected shelves have been physically visited but target not found!")
			# Reset visited flags for shelves we visited but weren't our target
			for shelf in self.detected_shelves:
				# Only keep visited=True for shelves we successfully identified
				if not shelf.get('shelf_id_confirmed', False):
					shelf['visited'] = False
			
			# Try again with reset shelves
			unvisited_shelves = [shelf for shelf in self.detected_shelves 
								if not shelf.get('visited', False)]
			
			if not unvisited_shelves:
				self.logger.error("Still no unvisited shelves after reset!")
				return False
		
		# Log current state
		self.logger.info(f"=== NAVIGATION STATE ===")
		self.logger.info(f"Current position: ({self.buggy_pose_x:.2f}, {self.buggy_pose_y:.2f})")
		self.logger.info(f"Target shelf ID: {self.current_target_shelf_id}")
		self.logger.info(f"Successfully scanned shelves: {sorted(self.successfully_scanned_shelves)}")
		self.logger.info(f"Unvisited shelves: {len(unvisited_shelves)}")
		
		# Determine which angle to use
		if self.current_target_shelf_id == 1:
			# Looking for shelf 1: use initial_angle from origin
			target_angle = self.initial_angle
			self.logger.info(f"Using initial angle: {target_angle}°")
			best_shelf = self.find_shelf_by_angle_from_spawn(unvisited_shelves, target_angle)
		else:
			# Looking for shelves 2, 3, 4: use angle from previous shelf's QR
			if hasattr(self, 'next_shelf_angle') and self.current_target_shelf_id - 1 in self.successfully_scanned_shelves:
				target_angle = self.next_shelf_angle
				self.logger.info(f"Using QR angle from shelf {self.current_target_shelf_id - 1}: {target_angle}°")
				
				# SIMPLIFIED: Use the stored last_confirmed_shelf_center directly
				if hasattr(self, 'last_confirmed_shelf_center') and self.last_confirmed_shelf_center:
					previous_shelf_center = self.last_confirmed_shelf_center
					self.logger.info(f"Using stored previous shelf center: {previous_shelf_center}")
					best_shelf = self.find_shelf_by_angle_from_position(unvisited_shelves, target_angle, previous_shelf_center)
				else:
					self.logger.error(f"No stored previous shelf center!")
					return False
			else:
				self.logger.error(f"No valid angle for shelf {self.current_target_shelf_id}!")
				return False
		
		if best_shelf:
			self.logger.info(f"→ Navigating to shelf at {best_shelf['center']}")
			self.start_shelf_inspection(best_shelf)
			return True
		
		self.logger.error(f"Could not find a suitable shelf")
		return False

	def navigate_to_initial_shelf(self):
		"""
		Navigate to the first shelf (Shelf 1).
		"""
		if not self.detected_shelves:
			self.logger.warn("Cannot select initial shelf: No shelves detected yet.")
			return False
		
		# Check if we've already visited shelf 1
		if 1 in self.visited_shelves:
			self.logger.info("Shelf 1 already visited, looking for next shelf")
			return self.navigate_to_next_shelf()
		
		unvisited_shelves = [s for s in self.detected_shelves if not s.get('visited', False)]
		if not unvisited_shelves:
			self.logger.info("All shelves have already been visited.")
			return False

		# Use initial angle to find what we hope is Shelf 1
		best_shelf = None
		min_angle_diff = float('inf')

		current_pos = (self.buggy_pose_x, self.buggy_pose_y)
		self.logger.info(f"Looking for Shelf 1 using initial angle {self.initial_angle}°")

		for shelf in unvisited_shelves:
			shelf_pos = shelf['center']
			
			# Calculate angle from current position to shelf
			angle_to_shelf = math.degrees(
				self.create_yaw_from_vector(
					shelf_pos[0], shelf_pos[1],
					current_pos[0], current_pos[1]
				)
			)
			
			# Normalize angles to [0, 360)
			normalized_shelf_angle = (angle_to_shelf + 360) % 360
			normalized_target_angle = (self.initial_angle + 360) % 360

			# Calculate the shortest angle difference
			angle_diff = abs(normalized_shelf_angle - normalized_target_angle)
			if angle_diff > 180:
				angle_diff = 360 - angle_diff
			
			self.logger.info(f"  -> Checking shelf at ({shelf_pos[0]:.2f}, {shelf_pos[1]:.2f}). "
						f"Angle: {normalized_shelf_angle:.1f}°. Difference: {angle_diff:.1f}°.")

			if angle_diff < min_angle_diff:
				min_angle_diff = angle_diff
				best_shelf = shelf
		
		if best_shelf:
			selected_pos = best_shelf['center']
			self.logger.info(f"Selected shelf at ({selected_pos[0]:.2f}, {selected_pos[1]:.2f}) "
						f"as potential Shelf 1 (angle diff: {min_angle_diff:.1f}°)")
			self.start_shelf_inspection(best_shelf)
			return True
		
		self.logger.error("CRITICAL: Could not determine an initial shelf to navigate to.")
		return False

	
	def start_shelf_inspection(self, shelf):
		"""Start inspection with fixed object distance and QR scanning positions."""
		self.current_inspection_shelf = shelf
		self.goal_queue = []
		center_x, center_y = shelf['center']
		orientation = shelf['orientation']  # This is the angle of the LONG side

		# --- Step 1: Calculate Object Scanning Positions (Long Sides) ---
		obj_distance = 2.9 # Fixed distance for object scanning
		obj_positions = []
		
		# To scan the long sides, we approach from a direction PERPENDICULAR to the shelf's orientation.
		self.logger.info("Calculating positions to approach the LONG sides for OBJECT scanning.")
		
		# Scan both long sides
		for side_offset in [math.pi / 2, -math.pi / 2]:  # +90 and -90 degrees
			approach_angle = orientation + side_offset
			pos = (center_x + obj_distance * math.cos(approach_angle),
				center_y + obj_distance * math.sin(approach_angle))
			
			# Face towards the shelf center
			face_angle = math.atan2(center_y - pos[1], center_x - pos[0])
			
			obj_positions.append({
				'pos': pos,
				'yaw': face_angle,
				'type': 'object'
			})

		# --- Step 2: Calculate QR Scanning Positions (Short Sides) ---
		qr_distance = 2.0  # Fixed distance for QR scanning
		qr_positions = []
		
		# To scan the short sides, we approach from a direction PARALLEL to the shelf's orientation.
		self.logger.info("Calculating positions to approach the SHORT sides for QR scanning.")
		for side_offset in [0, math.pi]:  # 0 and 180 degrees
			approach_angle = orientation + side_offset
			pos = (center_x + qr_distance * math.cos(approach_angle),
				center_y + qr_distance * math.sin(approach_angle))
			
			# Face towards the shelf center
			face_angle = math.atan2(center_y - pos[1], center_x - pos[0])
			
			qr_positions.append({
				'pos': pos, 
				'yaw': face_angle, 
				'type': 'qr'
			})
		
		# --- Step 3: Create the Goal Queue (Objects FIRST, then QR) ---
		self.goal_queue = obj_positions + qr_positions
		
		self.logger.info(f"Starting shelf inspection with fixed distances.")
		self.logger.info(f"  Object scanning: 2 positions at {obj_distance}m")
		self.logger.info(f"  QR scanning: 2 positions at {qr_distance}m")
		
		for i, pos_info in enumerate(self.goal_queue):
			pos = pos_info['pos']
			self.logger.info(f"  Position {i+1} ({pos_info['type']}): ({pos[0]:.2f}, {pos[1]:.2f})")
		
		self.process_goal_queue()
	def process_goal_queue(self):
		"""Process the next goal in the queue."""
		if not self.goal_queue:
			self.logger.info("Goal queue empty, finishing shelf inspection")
			self.finish_shelf_inspection()
			return
		
		# Don't process next goal if currently scanning
		if self.scanning_in_progress:
			self.logger.info("Scanning in progress, waiting...")
			return
		
		self.logger.info(f"Goal queue {self.goal_queue}")
		# FIXED: Get next goal WITHOUT popping it yet
		next_goal = self.goal_queue[0]  # Just peek at the first goal
		pos = next_goal['pos']
		yaw = next_goal['yaw']
		goal_type = next_goal['type']
		
		# Determine which side we're scanning for object detection
		if goal_type == 'object':
			# First object position is side A, second is side B
			remaining_object_goals = [g for g in self.goal_queue if g['type'] == 'object']
			if len(remaining_object_goals) == 2:
				self.current_side = 'A'
			else:
				self.current_side = 'B'
			self.logger.info(f"Preparing to scan Side {self.current_side}")
		
		self.inspection_stage = f"{goal_type}_scan"
		self.logger.info(f"Setting inspection stage to: {self.inspection_stage}")
		
		# Validate goal position is reachable
		if self.is_position_valid(pos):
			goal = self.create_goal_from_world_coord(pos[0], pos[1], yaw)
			
			if self.send_goal_from_world_pose(goal):
				self.logger.info(f"Moving to {goal_type} scanning position: ({pos[0]:.2f}, {pos[1]:.2f})")
				# FIXED: Don't pop here - let goal_result_callback handle it on success
			else:
				self.logger.error(f"Failed to send {goal_type} goal")
				self.handle_navigation_failure()
		else:
			self.logger.warn(f"Invalid {goal_type} position, skipping")
			# FIXED: Pop the invalid goal and try next one
			self.goal_queue.pop(0)
			self.process_goal_queue()  # Try next goal
	def finish_shelf_inspection(self):
		"""
		Finalizes inspection for the current shelf after all scanning goals are done.
		It publishes one aggregated message with all collected object and QR data.
		"""
		if self.current_inspection_shelf:
			self.logger.info(f"All scanning tasks completed for shelf at {self.current_inspection_shelf['center']}.")

			# Perform intelligent averaging of both sides
			self.intelligent_average_sides()
			
			# --- CRITICAL CHANGE: Only publish if QR code was found ---
			if self.qr_code_str != "Empty":
				# --- Build the message from the accumulator ---
				final_shelf_data = WarehouseShelf()
				
				# Convert the accumulated dictionary back into two lists for the message.
				if self.accumulated_objects:
					final_shelf_data.object_name = list(self.accumulated_objects.keys())
					final_shelf_data.object_count = [int(v) for v in self.accumulated_objects.values()]
				
				# Add the QR code string.
				final_shelf_data.qr_decoded = self.qr_code_str

				# Now publish the complete, aggregated message.
				self.publisher_shelf_data.publish(final_shelf_data)
				self.logger.info("--- Published Final Data for Shelf ---")
				self.logger.info(f"  Objects: {final_shelf_data.object_name} Counts: {final_shelf_data.object_count}")
				self.logger.info(f"  QR Code: '{final_shelf_data.qr_decoded}'")

				try:
					shelf_id = int(self.qr_code_str.split('_')[0])
					self.visited_shelves.add(shelf_id)
					
					# CRITICAL FIX: Update the shelf in the master list
					if shelf_id == self.current_target_shelf_id - 1:
						# Find and update the shelf in self.detected_shelves
						current_center = self.current_inspection_shelf['center']
						for shelf in self.detected_shelves:
							if shelf['center'] == current_center:
								shelf['visited'] = True
								shelf['shelf_id_confirmed'] = True
								self.last_confirmed_shelf_center = shelf['center']
								self.logger.info(f"Updated shelf {shelf_id} in master list at center: {self.last_confirmed_shelf_center}")
								break
						else:
							self.logger.error(f"Could not find shelf in detected_shelves to update!")
					
					self.logger.info(f"Shelf {shelf_id} successfully inspected and marked as visited.")
				except (ValueError, IndexError):
					self.logger.error("Could not parse shelf ID from QR code to mark as visited.")
			else:
				self.logger.warn("Shelf inspection finished, but NO QR CODE was found. Data will not be published.")
				# Still mark as visited to avoid getting stuck
				current_center = self.current_inspection_shelf['center']
				for shelf in self.detected_shelves:
					if shelf['center'] == current_center:
						shelf['visited'] = True
						break

		# Reset state variables to prepare for the next high-level action.
		self.current_inspection_shelf = None
		self.inspection_stage = 'idle'
		self.goal_queue = []
		self.qr_code_str = "Empty"
		# Clear accumulated objects and best frames for next shelf
		self.accumulated_objects = {}
		self.best_frame_side_a = None
		self.best_frame_side_b = None
		self.side_a_frames = []
		self.side_b_frames = []
		
	def publish_shelf_data(self):
		"""Publish shelf data only for correctly identified shelves."""
		if hasattr(self, 'shelf_objects_curr') and self.qr_code_str != "Empty":
			try:
				shelf_id = int(self.qr_code_str.split('_')[0])
				
				# Only publish if this was our target shelf
				if shelf_id in self.successfully_scanned_shelves:
					shelf_data = WarehouseShelf()
					shelf_data.object_name = self.shelf_objects_curr.object_name
					shelf_data.object_count = self.shelf_objects_curr.object_count
					shelf_data.qr_decoded = self.qr_code_str
					
					self.publisher_shelf_data.publish(shelf_data)
					
					self.logger.info(f"✓ Published data for shelf {shelf_id}:")
					self.logger.info(f"  Objects: {list(zip(shelf_data.object_name, shelf_data.object_count))}")
					self.logger.info(f"  QR: {shelf_data.qr_decoded}")
			except:
				self.logger.error("Error publishing shelf data")

	def update_gui_display(self, shelf_data):
		"""Update GUI display with shelf data."""
		if PROGRESS_TABLE_GUI:
			# Display objects
			obj_str = ""
			for name, count in zip(shelf_data.object_name, shelf_data.object_count):
				obj_str += f"{name}: {count}\n"
			
			box_app.change_box_text(0, self.table_col_count, obj_str)
			box_app.change_box_color(0, self.table_col_count, "cyan")
			
			# Display QR code
			box_app.change_box_text(1, self.table_col_count, shelf_data.qr_decoded)
			box_app.change_box_color(1, self.table_col_count, "yellow")
			
			self.table_col_count += 1


	def handle_navigation_failure(self):
		"""Handle navigation failures with recovery logic."""
		self.recovery_attempts += 1
		
		# Proactively cancel any lingering goal handle to ensure a clean state.
		self.cancel_current_goal()

		# CRITICAL FIX: Immediately reset state to allow a new recovery goal to be sent.
		self.goal_completed = True
		self.goal_handle_curr = None
		
		if self.recovery_attempts < self.max_recovery_attempts:
			self.logger.info(f"Navigation failed, attempting recovery {self.recovery_attempts}")
			
			# **NEW**: Try to move around obstacle
			self.attempt_obstacle_avoidance()
			
		else:
			self.logger.warn("Max recovery attempts reached, skipping current goal")
			self.recovery_attempts = 0
			
			# FIXED: Pop the failed goal before trying next one
			if self.goal_queue:
				failed_goal = self.goal_queue.pop(0)
				self.logger.info(f"Skipping failed {failed_goal['type']} goal")
			
			# Skip current goal and try next one
			if self.goal_queue:
				self.process_goal_queue()
			else:
				self.finish_shelf_inspection()

	def attempt_obstacle_avoidance(self):
		"""Simple obstacle avoidance by moving to a nearby position."""
		if not self.current_inspection_shelf:
			return
		
		# Calculate alternative position slightly offset from original
		center_x, center_y = self.current_inspection_shelf['center']
		
		# Try positions at different angles around the shelf
		for angle_offset in [0.5, -0.5, 1.0, -1.0]:  # radians
			alt_angle = self.current_inspection_shelf['orientation'] + angle_offset
			alt_pos = (center_x + 1.5 * math.cos(alt_angle),
					center_y + 1.5 * math.sin(alt_angle))
			
			goal = self.create_goal_from_world_coord(alt_pos[0], alt_pos[1])
			if self.send_goal_from_world_pose(goal):
				self.logger.info("Attempting obstacle avoidance")
				break
			
	def _start_scan_after_delay(self):
		"""
		This function is called by a one-shot timer. It cancels the timer
		and then begins the actual scanning process.
		"""
		self.logger.info("Delay complete. Now starting timed object scan.")
		
		# Cancel the timer that called this function to make it a one-shot action
		if self.scan_delay_timer is not None:
			self.scan_delay_timer.cancel()
			self.scan_delay_timer = None
		
		# Now, call the original function to start scanning
		self.start_scanning_at_position()

	def is_position_valid(self, pos):
		"""**NEW**: Check if a position is valid for navigation."""
		x, y = pos
		
		# **NEW**: Basic bounds checking
		if abs(x) > 50 or abs(y) > 50:  # Reasonable map bounds
			return False
			
		# **NEW**: Check against map if available
		if self.simple_map_curr:
			map_info = self.simple_map_curr.info
			map_x, map_y = self.get_map_coord_from_world_coord(x, y, map_info)
			
			# Check if within map bounds
			if (map_x < 0 or map_x >= map_info.width or 
				map_y < 0 or map_y >= map_info.height):
				return False
				
			# Check if position is free space
			height, width = map_info.height, map_info.width
			map_array = np.array(self.simple_map_curr.data).reshape((height, width))
			
			if map_array[map_y, map_x] != 0:  # Not free space
				return False
		
		return True
	def start_scanning_at_position(self):
		"""Enhanced scanning with per-side frame capture."""
		self.scanning_in_progress = True
		self.scanning_start_time = time.time()
		self.qr_found_at_position = False
		self.current_scan_type = 'qr' if self.inspection_stage == 'qr_scan' else 'object'
		
		if self.current_scan_type == 'object':
			# Reset frame capture for current side
			self.frame_capture_count = 0
			
			# Clear frames for current side only
			if self.current_side == 'A':
				self.side_a_frames = []
				self.logger.info(f"Starting Side A object scanning with {self.frames_per_position} frame captures")
			else:
				self.side_b_frames = []
				self.logger.info(f"Starting Side B object scanning with {self.frames_per_position} frame captures")
			
			# Start micro-rotation
			self.start_micro_rotation()
			
			# Start frame capture timer
			self.frame_capture_timer = self.create_timer(
				self.frame_capture_interval, 
				self.capture_frame
			)
		else:
			# QR scanning remains the same
			self.logger.info(f"Starting QR scanning for {self.scanning_duration} seconds")
			self.scan_timer = self.create_timer(self.scanning_duration, self.finish_scanning)

	def finish_scanning(self):
		"""Finish scanning at current position."""
		if not self.scanning_in_progress:
			return
		self.scanning_in_progress = False

		# Cancel any active timers
		if self.scan_timer is not None:
			self.scan_timer.cancel()
			self.scan_timer = None
		
		if self.frame_capture_timer is not None:
			self.frame_capture_timer.cancel()
			self.frame_capture_timer = None
		
		# Stop micro rotation if active
		if hasattr(self, 'micro_rotation_timer') and self.micro_rotation_timer:
			self.micro_rotation_timer.cancel()
			# Stop the rover
			self.rover_move_manual_mode(0.0, 0.0)

		if self.current_scan_type == 'qr' and self.qr_found_at_position:
			self.logger.info("QR code found during scan!")
			
			# Process based on whether it's the correct shelf
			try:
				if self.qr_code_str != "Empty":
					shelf_id = int(self.qr_code_str.split('_')[0])
					if shelf_id == self.current_target_shelf_id:
						# Correct shelf - we're done!
						self.logger.info(f"✓ Found target shelf {shelf_id}!")
						self.finish_shelf_inspection()
					else:
						# Wrong shelf - finish and move on
						self.logger.warn(f"✗ Found shelf {shelf_id} instead of target {self.current_target_shelf_id}")
						self.finish_shelf_inspection()
			except:
				self.logger.error("Error parsing shelf ID")
				self.process_goal_queue()
		else:
			if self.current_scan_type == 'object':
				self.logger.info("Object scanning completed")
				self.process_object_data()
			else:
				self.logger.warn("No QR found at this position")
			
			self.process_goal_queue()
	def retry_shelf_if_needed(self):
		"""Retry shelf if QR code wasn't found."""
		if (self.current_inspection_shelf and 
			not self.current_inspection_shelf.get('visited', False) and
			not self.goal_queue):
			
			self.logger.info("Retrying shelf inspection with different approach angles")
			
			# Try different approach angles
			center_x, center_y = self.current_inspection_shelf['center']
			orientation = self.current_inspection_shelf['orientation']
			
			# Additional QR scanning positions with different angles
			retry_positions = []
			# Try closer positions on short sides
			for distance in [1.0, 1.5]:  # Closer distances
				for side_offset in [-math.pi/2, math.pi/2]:  # Short sides only
					approach_angle = orientation + side_offset
					pos = (center_x + distance * math.cos(approach_angle),
						center_y + distance * math.sin(approach_angle))
					
					# FIXED: Face towards the shelf center
					face_angle = math.atan2(center_y - pos[1], center_x - pos[0])
					
					retry_positions.append({
						'pos': pos,
						'yaw': face_angle,
						'type': 'qr'
					})
			
			# **ADDITIONAL**: Try slight angle variations on short sides
			for angle_var in [-0.3, 0.3]:  # Small angle variations (about 17 degrees)
				for side_offset in [-math.pi/2, math.pi/2]:
					approach_angle = orientation + side_offset + angle_var
					pos = (center_x + 2.0 * math.cos(approach_angle),
						center_y + 2.0 * math.sin(approach_angle))
					
					# FIXED: Face towards the shelf center
					face_angle = math.atan2(center_y - pos[1], center_x - pos[0])
					
					retry_positions.append({
						'pos': pos,
						'yaw': face_angle,
						'type': 'qr'
					})
			
			self.goal_queue = retry_positions
			self.logger.info(f"Added {len(retry_positions)} retry QR scanning positions")
			self.process_goal_queue()
			return True
		
		return False


	def process_object_data(self):
		"""Process and publish object data."""
		if hasattr(self, 'shelf_objects_curr'):
			# Add QR code to object data
			# self.shelf_objects_curr.qr_decoded = self.qr_code_str
			# self.publisher_shelf_data.publish(self.shelf_objects_curr)
			self.logger.info(f"Published shelf data: {len(self.shelf_objects_curr.object_name)} objects, QR: {self.qr_code_str}")
	def get_next_shelf_id_to_visit(self):
		"""Determine the next shelf ID we should visit in sequence."""
		# Find the lowest shelf ID we haven't visited yet
		for shelf_id in range(1, self.shelf_count + 1):
			if shelf_id not in self.visited_shelves:
				return shelf_id
		return None  # All shelves visited
	def find_shelf_by_angle(self, shelves, target_angle):
		"""Find the shelf that best matches the target angle from current position."""
		best_shelf = None
		min_angle_diff = float('inf')
		
		current_pos = (self.buggy_pose_x, self.buggy_pose_y)
		
		self.logger.info(f"Finding shelf from position ({current_pos[0]:.2f}, {current_pos[1]:.2f}) with target angle {target_angle}°")
		
		for shelf in shelves:
			shelf_pos = shelf['center']
			
			# Calculate angle from current position to shelf
			angle_to_shelf_rad = self.create_yaw_from_vector(
				shelf_pos[0], shelf_pos[1],
				current_pos[0], current_pos[1]
			)
			angle_to_shelf_deg = math.degrees(angle_to_shelf_rad)
			
			# Normalize angles to [0, 360)
			angle_to_shelf_normalized = (angle_to_shelf_deg + 360) % 360
			target_angle_normalized = (target_angle + 360) % 360
			
			# Calculate the shortest angle difference
			angle_diff = abs(angle_to_shelf_normalized - target_angle_normalized)
			if angle_diff > 180:
				angle_diff = 360 - angle_diff
			
			self.logger.info(f"  Shelf at ({shelf_pos[0]:.2f}, {shelf_pos[1]:.2f}): "
						f"angle={angle_to_shelf_normalized:.1f}°, diff={angle_diff:.1f}°")
			
			if angle_diff < min_angle_diff:
				min_angle_diff = angle_diff
				best_shelf = shelf
		
		if best_shelf:
			self.logger.info(f"Best match: shelf at {best_shelf['center']} (angle diff: {min_angle_diff:.1f}°)")
		
		return best_shelf
	def find_closest_shelf(self, shelves):
		"""Find the closest shelf to current position."""
		best_shelf = None
		min_distance = float('inf')
		
		current_pos = (self.buggy_pose_x, self.buggy_pose_y)
		
		for shelf in shelves:
			shelf_pos = shelf['center']
			distance = euclidean(current_pos, shelf_pos)
			
			if distance < min_distance:
				min_distance = distance
				best_shelf = shelf
		
		if best_shelf:
			self.logger.info(f"Closest shelf at {best_shelf['center']} (distance: {min_distance:.2f}m)")
		
		return best_shelf
	def find_shelf_by_angle_from_spawn(self, shelves, target_angle):
		"""Find shelf using angle FROM SPAWN/ORIGIN POINT, not current position."""
		best_shelf = None
		min_angle_diff = float('inf')
		
		# Use spawn point or origin for consistent angle calculation
		spawn_point = (0.0, 0.0)  # Use origin for consistency
		
		self.logger.info(f"Finding shelf with target angle {target_angle}° from origin (0, 0)")
		
		for shelf in shelves:
			shelf_pos = shelf['center']
			
			# Calculate angle from SPAWN/ORIGIN to shelf (not from current position!)
			angle_to_shelf_rad = math.atan2(shelf_pos[1] - spawn_point[1], 
											shelf_pos[0] - spawn_point[0])
			angle_to_shelf_deg = math.degrees(angle_to_shelf_rad)
			
			# Normalize angles to [0, 360)
			angle_to_shelf_normalized = (angle_to_shelf_deg + 360) % 360
			target_angle_normalized = (target_angle + 360) % 360
			
			# Calculate the shortest angle difference
			angle_diff = abs(angle_to_shelf_normalized - target_angle_normalized)
			if angle_diff > 180:
				angle_diff = 360 - angle_diff
			
			self.logger.info(f"  Shelf at ({shelf_pos[0]:.2f}, {shelf_pos[1]:.2f}): "
						f"angle from origin={angle_to_shelf_normalized:.1f}°, diff={angle_diff:.1f}°")
			
			if angle_diff < min_angle_diff:
				min_angle_diff = angle_diff
				best_shelf = shelf
		
		if best_shelf:
			self.logger.info(f"✓ Best match: shelf at {best_shelf['center']} (angle diff: {min_angle_diff:.1f}°)")
		
		return best_shelf
	def find_shelf_by_angle_from_position(self, shelves, target_angle, reference_pos):
		"""Find shelf using angle from a specific reference position (e.g., previous shelf)."""
		best_shelf = None
		min_angle_diff = float('inf')
		
		self.logger.info(f"Finding shelf with target angle {target_angle}° from position ({reference_pos[0]:.2f}, {reference_pos[1]:.2f})")
		
		for shelf in shelves:
			shelf_pos = shelf['center']
			
			# Calculate angle from reference position to shelf
			angle_to_shelf_rad = math.atan2(shelf_pos[1] - reference_pos[1], 
											shelf_pos[0] - reference_pos[0])
			angle_to_shelf_deg = math.degrees(angle_to_shelf_rad)
			
			# Normalize angles to [0, 360)
			angle_to_shelf_normalized = (angle_to_shelf_deg + 360) % 360
			target_angle_normalized = (target_angle + 360) % 360
			
			# Calculate the shortest angle difference
			angle_diff = abs(angle_to_shelf_normalized - target_angle_normalized)
			if angle_diff > 180:
				angle_diff = 360 - angle_diff
			
			self.logger.info(f"  Shelf at ({shelf_pos[0]:.2f}, {shelf_pos[1]:.2f}): "
						f"angle from reference={angle_to_shelf_normalized:.1f}°, diff={angle_diff:.1f}°")
			
			if angle_diff < min_angle_diff:
				min_angle_diff = angle_diff
				best_shelf = shelf
		
		if best_shelf:
			self.logger.info(f"✓ Best match: shelf at {best_shelf['center']} (angle diff: {min_angle_diff:.1f}°)")
		
		return best_shelf
	def start_micro_rotation(self):
		"""Perform small left-right rotation during object scanning."""
		# Create a timer for micro movements
		self.micro_rotation_timer = self.create_timer(2.0, self.toggle_micro_rotation)
		self.micro_rotation_direction = 1  # 1 for left, -1 for right
	def toggle_micro_rotation(self):
		"""Toggle rotation direction for better coverage."""
		if self.scanning_in_progress and self.current_scan_type == 'object':
			# Send small rotation command
			self.rover_move_manual_mode(0.0, 0.1 * self.micro_rotation_direction)
			self.micro_rotation_direction *= -1  # Toggle direction
		else:
			# Stop micro rotation and rover movement
			self.rover_move_manual_mode(0.0, 0.0)
			if hasattr(self, 'micro_rotation_timer'):
				self.micro_rotation_timer.cancel()
	def capture_frame(self):
		"""Capture a frame during object scanning."""
		if not self.scanning_in_progress or self.current_scan_type != 'object':
			return
		
		self.frame_capture_count += 1
		self.logger.info(f"Side {self.current_side} - Capturing frame {self.frame_capture_count}/{self.frames_per_position}")
		
		# Check if we've captured enough frames
		if self.frame_capture_count >= self.frames_per_position:
			# Cancel the frame capture timer
			if self.frame_capture_timer:
				self.frame_capture_timer.cancel()
				self.frame_capture_timer = None
			
			# Select best frame for current side
			self.select_best_frame_for_side()
			self.finish_scanning()
	def select_best_frame(self):
		"""Select the frame with the most objects detected."""
		if not self.frame_captures:
			self.logger.warn("No frames captured during object scanning")
			return
		
		# Find the frame with the maximum number of objects
		best_frame = max(self.frame_captures, key=lambda x: x['total_objects'])
		
		self.logger.info(f"Best frame selected with {best_frame['total_objects']} total objects")
		
		# Update accumulated objects with the best frame
		self.accumulated_objects = {}
		for name, count in zip(best_frame['object_name'], best_frame['object_count']):
			self.accumulated_objects[name] = count
		
		# Log the selected objects
		self.logger.info(f"Selected objects: {self.accumulated_objects}")
	def select_best_frame_for_side(self):
		"""Select the best frame for the current side being scanned."""
		if self.current_side == 'A':
			frames = self.side_a_frames
		else:
			frames = self.side_b_frames
		
		if not frames:
			self.logger.warn(f"No frames captured for Side {self.current_side}")
			return
		
		# Find the frame with the maximum number of objects (up to max limit)
		best_frame = max(frames, key=lambda x: x['total_objects'])
		
		self.logger.info(f"Side {self.current_side} - Best frame selected with {best_frame['total_objects']} total objects")
		
		# Store the best frame
		if self.current_side == 'A':
			self.best_frame_side_a = best_frame
		else:
			self.best_frame_side_b = best_frame
			
		# Log the selected objects for this side
		objects_str = ", ".join([f"{name}:{count}" for name, count in 
							zip(best_frame['object_name'], best_frame['object_count'])])
		self.logger.info(f"Side {self.current_side} best frame objects: {objects_str}")
	def intelligent_average_sides(self):
		"""Intelligently select the higher count from both sides."""
		if not self.best_frame_side_a or not self.best_frame_side_b:
			self.logger.warn("Missing best frames from one or both sides")
			# Use whatever we have
			if self.best_frame_side_a:
				self.accumulated_objects = dict(zip(self.best_frame_side_a['object_name'], 
												self.best_frame_side_a['object_count']))
			elif self.best_frame_side_b:
				self.accumulated_objects = dict(zip(self.best_frame_side_b['object_name'], 
												self.best_frame_side_b['object_count']))
			else:
				self.accumulated_objects = {}
			return
		
		# Create dictionaries for easy lookup
		side_a_objects = dict(zip(self.best_frame_side_a['object_name'], 
								self.best_frame_side_a['object_count']))
		side_b_objects = dict(zip(self.best_frame_side_b['object_name'], 
								self.best_frame_side_b['object_count']))
		
		# Combine all unique object names
		all_objects = set(side_a_objects.keys()) | set(side_b_objects.keys())
		
		self.accumulated_objects = {}
		
		for obj_name in all_objects:
			count_a = side_a_objects.get(obj_name, 0)
			count_b = side_b_objects.get(obj_name, 0)
			
			if count_a > 0 and count_b > 0:
				# Object seen on both sides - take the higher count
				final_count = max(count_a, count_b)
				self.logger.info(f"{obj_name}: Side A={count_a}, Side B={count_b}, Final={final_count} (higher)")
			elif count_a > 0:
				# Only seen on side A
				final_count = count_a
				self.logger.info(f"{obj_name}: Only on Side A={count_a}, Final={final_count}")
			else:
				# Only seen on side B
				final_count = count_b
				self.logger.info(f"{obj_name}: Only on Side B={count_b}, Final={final_count}")
			
			if final_count > 0:
				self.accumulated_objects[obj_name] = final_count
		
		# Verify total doesn't exceed maximum
		total_objects = sum(self.accumulated_objects.values())
		if total_objects > self.max_objects_per_shelf:
			self.logger.warn(f"Total objects {total_objects} exceeds maximum {self.max_objects_per_shelf}")
			# You might want to implement a strategy to reduce counts here
		
		self.logger.info(f"Final objects with higher counts: {self.accumulated_objects}")
def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()