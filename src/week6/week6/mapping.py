import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose, Twist, Quaternion, PoseStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

import threading
import cv2
import numpy as np

class MapWithPose(Node):
    def __init__(self):
        super().__init__('mapping')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.is_init_pose = False
        self.pose = None
        self.goal_start = False
        self.map = None

    def pose_callback(self, msg):
        """
        주기적으로 로봇의 현재 위치를 받아옵니다. 초기 위치를 기록합니다.
        """
        if not self.is_init_pose:
            self.get_logger().info('Received initial pose.')
            self.init_pose = msg.pose.pose
            self.is_init_pose = True

        self.pose = msg.pose.pose
    
    def map_callback(self, msg):
        """
        맵 데이터를 처리하고 목표 지점을 설정합니다.
        """
        if self.pose is None:
            self.get_logger().info('No pose received yet.')
            return

        if msg is None:
            self.get_logger().warn('No map received.')
            return

        if self.goal_start:
            return  # 목표 지점을 설정한 이후에는 추가 작업 수행 안 함

        self.map = msg
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.data = msg.data

        map_img = self.data_to_image(self.data)
        points = self.find_boundary(map_img)

        if points is None:
            self.get_logger().info('Mapping complete. No more boundaries detected.')
            self.finish_mapping()
            return

        goal_point = self.find_goal(points)
        if goal_point:
            self.pub_goal(goal_point)
        else:
            self.get_logger().warn('No valid goal point detected.')

    def pub_goal(self, goal_point):
        """
        목표 지점을 ROS 메시지로 퍼블리시합니다.
        """
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_point[0]
        goal_pose.pose.position.y = goal_point[1]
        self.get_logger().info(f'Published Goal Point: {goal_point}')
        self.goal_pose_pub.publish(goal_pose)

    def find_goal(self, points):
        """
        경계선 점 중에서 안전한 목표 지점을 찾습니다.
        """
        goal_point = None
        min_distance = float('inf')

        for point in points:
            world_point = self.map_to_world(point)
            if self.distance(world_point) < min_distance and self.is_goal_safe(world_point[0], world_point[1]):
                goal_point = world_point
                min_distance = self.distance(world_point)

        if goal_point:
            return goal_point

        self.get_logger().warn('No goal found in safe areas. Expanding search.')
        for point in points:
            world_point = self.map_to_world(point)
            if self.distance(world_point) < min_distance:
                goal_point = world_point
                min_distance = self.distance(world_point)

        return goal_point
    
    def map_to_world(self, point):
        """
        맵 픽셀 좌표를 월드 좌표로 변환합니다.
        """
        x_pixel, y_pixel = point
        x_world = float(x_pixel * self.resolution + self.origin.position.x)
        y_world = float(y_pixel * self.resolution + self.origin.position.y)
        return x_world, y_world

    def distance(self, point):
        """
        현재 위치와 목표 지점 간의 유클리드 거리 계산.
        """
        if self.pose is None:
            return float('inf')
        x = self.pose.position.x
        y = self.pose.position.y
        x1, y1 = point
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    def data_to_image(self, data):
        """
        맵 데이터를 2D 이미지로 변환합니다.
        """
        img = np.array(data).reshape(self.height, self.width)
        img = img.astype(np.uint8)
        return img

    def find_boundary(self, image):
        """
        맵의 경계선을 찾습니다.
        """
        diff_x = image[:, 1:] - image[:, :-1]
        diff_y = image[1:, :] - image[:-1, :]

        boundary_x = ((image[:, :-1] == 0) & (diff_x == -1)).astype(np.uint8) * 255
        boundary_y = ((image[:-1, :] == 0) & (diff_y == -1)).astype(np.uint8) * 255

        binary_edges = np.zeros_like(image, dtype=np.uint8)
        binary_edges[:, 1:] = boundary_x
        binary_edges[1:, :] += boundary_y

        points = np.where(binary_edges == 255)
        if len(points[0]) == 0:
            return None

        return list(zip(points[1], points[0]))

    def is_goal_safe(self, goal_x, goal_y, safety_radius=0.4):
        """
        목표 지점 주변이 안전한지 확인합니다.
        """
        map_x = int((goal_x - self.origin.position.x) / self.resolution)
        map_y = int((goal_y - self.origin.position.y) / self.resolution)

        radius_pixels = int(safety_radius / self.resolution)
        width, height = self.width, self.height

        for y in range(map_y - radius_pixels, map_y + radius_pixels + 1):
            for x in range(map_x - radius_pixels, map_x + radius_pixels + 1):
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                if self.map.data[y * width + x] == 100:
                    return False
        return True

    def finish_mapping(self):
        """
        맵핑 완료 시 초기 위치로 이동합니다.
        """
        self.get_logger().info('Mapping finished. Returning to initial pose.')
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose = self.init_pose
        self.goal_pose_pub.publish(goal)
        self.goal_start = True


def main(args=None):
    rclpy.init()
    node = MapWithPose()

    max_wait_time = 3.0
    wait_start_time = node.get_clock().now().seconds_nanoseconds()[0]

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        current_time = node.get_clock().now().seconds_nanoseconds()[0]

        if node.is_init_pose and node.map:
            node.get_logger().info("Initial pose and map received. Starting...")
            break

        if current_time - wait_start_time > max_wait_time:
            node.get_logger().warn("Timeout: Initial pose or map not received.")
            return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        rclpy.shutdown()
