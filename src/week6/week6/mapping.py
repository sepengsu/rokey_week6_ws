import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Twist, Quaternion, PoseStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

import threading
import cv2, numpy as np


class MapWithPose(Node):
    def __init__(self):
        super().__init__('mapping')
        self.map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.is_init_pose = False
        self.pose = None
        self.goal_start = False

    def pose_callback(self, msg):
        '''
        위치를 주기적으로 받습니다. 
        1. 초기 위치를 받아오게 합니다. 
        pose 의 포멧
          pose:
            position:
                x: -1.7216006539614395
                y: 1.1057414292027288
                z: 0.0
            orientation:
                x: 0.0
                y: 0.0
                z: 0.07625620835059867
                w: 0.9970882562180692

        '''
        if not self.is_init_pose:
            self.get_logger().info('Get initial pose')
            self.init_pose = msg.pose.pose
            self.is_init_pose = True

        self.pose = msg.pose.pose
    
    def map_callback(self, msg):
        '''
        맵을 받아옵니다.
        1. 맵을 받아옵니다.
        2. 맵에서 -1, 0 100을 각각 255, 0, 100으로 바꿔 이미지로 출력합니다.
        3. 맵에서 -1과 0의 경계선을 찾아 출력합니다
        4. 경계선을 찾아서 출력합니다.
        '''
        if self.pose is None:
            self.get_logger().info('No Pose')
            return
        if self.goal_start:
            return # 목표지점을 찾았으면 더 이상 맵을 받아오지 않습니다.
        self.map = msg
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.data = msg.data
        self.get_logger().info('Get map')
        # 2. 맵에서 -1, 0 100을 각각 이미지로 출력합니다.
        # -1은 

        map_img = self.data_to_image(self.data)
        points = self.find_boundary(map_img)
        if points is None: # 경계선이 없으면 맵핑이 끝난 것으로 판단합니다.
            self.get_logger().info('Mapping is done')
            self.finish_mapping()
        goal_point = self.find_goal(points)
        self.pub_goal(goal_point)
        

    def pub_goal(self, goal_point):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'  # 'map' 좌표계 사용
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_point[0]
        goal_pose.pose.position.y = goal_point[1]
        self.get_logger().info(f'Goal Point: {goal_point}')
        self.goal_pose_pub.publish(goal_pose)

    def find_goal(self, points):
        '''
        현재의 포즈와 points를 이용하여 goal point를 찾습니다.
        '''
        goal_point = None
        min_distance = 10000
        for point in points:
            point = self.map_to_world(point)
            if self.distance(point) < min_distance and self.is_goal_safe(point[0], point[1]):
                goal_point = point
                min_distance = self.distance(point)
        if goal_point:
            return goal_point
        self.get_logger().info('No detectable goal point in safe area')
        min_distance = 10000
        for point in points:
            point = self.map_to_world(point)
            if self.distance(point)< min_distance and self.is_goal_safe(point[0], point[1], safety_radius=0):
                goal_point = point
                min_distance = self.distance(point)

        if goal_point:
            return goal_point
        self.get_logger().info('No detectable goal point in unsafe area')
        return None
    
    def map_to_world(self, point):
        """
        맵 픽셀 좌표를 월드 좌표로 변환.
        :param point: (x_pixel, y_pixel)
        :return: (x_world, y_world)
        """
        x_pixel, y_pixel = point
        x_world = float(x_pixel * self.resolution + self.origin.position.x)
        y_world = float(y_pixel * self.resolution + self.origin.position.y)
        return x_world, y_world

    def distance(self,point):
        x = self.pose.position.x
        y = self.pose.position.y
        x1, y1 = point
        return np.sqrt((x-x1)**2 + (y-y1)**2)
    
    def data_to_image(self, data):
        '''
        맵 데이터를 이미지로 변환합니다.
        '''
        img = np.array(data).reshape(self.height, self.width)
        img = img.astype(np.uint8)
        return img
    
    def find_boundary(self, image):
        '''
        맵에서 경계선을 찾습니다.
        여기서 픽셀 값이 급격하게 바뀌는 구간 (0 to -1)을 찾고,
        주위 (3x3)에서 255가 없으면 해당 경계를 제거합니다.
        '''
        # 1. 경계 계산 (X, Y 방향에서의 차이)
        diff_x = image[:, 1:] - image[:, :-1]  # X 방향 차이
        diff_y = image[1:, :] - image[:-1, :]  # Y 방향 차이

        # 2. 0 → -1 변화를 찾아 경계로 설정
        boundary_x = ((image[:, :-1] == 0) & (diff_x == -1)).astype(np.uint8) * 255
        boundary_y = ((image[:-1, :] == 0) & (diff_y == -1)).astype(np.uint8) * 255

        # 3. 결과 병합
        binary_edges = np.zeros_like(image, dtype=np.uint8)
        binary_edges[:, 1:] = boundary_x  # X 방향
        binary_edges[1:, :] += boundary_y  # Y 방향

        points = np.where(binary_edges == 255)
        if len(points[0]) == 0:
            return None
        return list(zip(points[1], points[0]))
    
    def is_goal_safe(self, goal_x, goal_y,safety_radius=0.40):
        """
        목표 좌표 주변이 로봇이 통과 가능한지 확인합니다.
        :param map_data: 맵의 OccupancyGrid 데이터 (list).
        :param goal_x: 목표 좌표 X (월드 좌표계).
        :param goal_y: 목표 좌표 Y (월드 좌표계).
        :param resolution: 맵의 해상도 (m/pixel).
        :param origin: 맵의 원점 (Pose).
        :param robot_radius: 로봇 반경 (m).
        :return: True(안전) 또는 False(안전하지 않음).
        """
        # 월드 좌표 -> 맵 픽셀 좌표 변환
        map_x = int((goal_x - self.origin.position.x) / self.resolution)
        map_y = int((goal_y - self.origin.position.y) / self.resolution)
        
        # 로봇 반경 -> 픽셀 단위로 변환
        radius_pixels = int(safety_radius / self.resolution)
        width, height = self.width, self.height

        # 100인 값에 대해서 반경 만큼 100으로 채우기
        for y in range(map_y - radius_pixels, map_y + radius_pixels + 1): # y
            for x in range(map_x - radius_pixels, map_x + radius_pixels + 1): # x
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                if self.map.data[y * width + x] == 100: # 장애물이 있으면
                    return False
        return True  # 안전
    
    def finish_mapping(self):
        '''
        맵핑이 끝났을 때 수행할 작업을 수행합니다.
        '''
        self.get_logger().info('Finish mapping')
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose = self.init_pose
        self.goal_pose_pub.publish(goal)
        self.goal_start = True 
        
        
def main(args=None):
    rclpy.init()
    node = MapWithPose()
    while rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()


