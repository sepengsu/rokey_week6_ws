import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
import numpy as np
import cv2
class MapWithPose(Node):
    def __init__(self):
        super().__init__('mapping')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # 초기 변수 설정
        self.is_init_pose = False
        self.pose = None
        self.goal_start = False
        self.map = None
        self.count = 0

    def pose_callback(self, msg):
        """로봇의 초기 위치를 받아오는 콜백"""
        if not self.is_init_pose:
            self.get_logger().info('Received initial pose.')
            self.init_pose = msg.pose.pose
            self.is_init_pose = True

        self.pose = msg.pose.pose
    
    def map_callback(self, msg):
        """맵 데이터를 처리하고 목표 지점을 설정하는 콜백"""
        if not self.is_init_pose or self.pose is None:
            self.get_logger().info('Waiting for pose information...')
            return

        if msg is None or len(msg.data) == 0:
            self.get_logger().warn('Received empty map or no map data.')
            return

        if self.goal_start:
            return  # 목표 지점을 설정한 이후에는 추가 작업을 하지 않음

        # 맵 정보 처리
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.data = msg.data
        map_img = self.data_to_image(self.data)  # 맵 데이터를 이미지로 변환

        points = self.find_boundary(map_img)  # 경계선 찾기
        self.count += 1

        if points is None and not np.any(map_img == 255):
            if self.count > 10:
                self.get_logger().info('Mapping complete. No more boundaries detected.')
                self.finish_mapping()
                return
            return  # 경계선이 없으면 작업을 하지 않음    

        goal_point = self.find_goal(points)
        if goal_point:
            self.pub_goal(goal_point)
        else:
            self.get_logger().warn('No valid goal point detected.')


    def pub_goal(self, goal_point):
        """목표 지점을 퍼블리시하는 함수"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_point[0]
        goal_pose.pose.position.y = goal_point[1]
        self.get_logger().info(f'Published Goal Point: {goal_point}')
        self.goal_pose_pub.publish(goal_pose)

    def find_goal(self, points):
        """경계선 점 중에서 안전한 목표 지점을 찾는 함수"""
        goal_point = None
        min_distance = float('inf')

        # 첫 번째 시도: 안전한 목표 지점 찾기
        for point in points:
            world_point = self.map_to_world(point)
            if self.distance(world_point) < min_distance and self.is_goal_safe(world_point[0], world_point[1]):
                goal_point = world_point
                min_distance = self.distance(world_point)

        if not goal_point:
            self.get_logger().warn('No safe goal found in initial search. Expanding search...')
            # 두 번째 시도: 안전하지 않더라도 최소 거리인 목표 지점 선택
            min_distance = float('inf')
            for point in points:
                world_point = self.map_to_world(point)
                if self.distance(world_point) < min_distance and self.is_goal_safe(world_point[0], world_point[1]):
                    goal_point = world_point
                    min_distance = self.distance(world_point)

        return goal_point

    
    def map_to_world(self, point):
        """맵 픽셀 좌표를 월드 좌표로 변환하는 함수"""
        x_pixel, y_pixel = point
        x_world = float(x_pixel * self.resolution + self.origin.position.x)
        y_world = float(y_pixel * self.resolution + self.origin.position.y)
        return x_world, y_world

    def distance(self, point):
        """현재 위치와 목표 지점 간의 유클리드 거리 계산"""
        if self.pose is None:
            return float('inf')
        x = self.pose.position.x
        y = self.pose.position.y
        x1, y1 = point
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    def data_to_image(self, data):
        """맵 데이터를 2D 이미지로 변환하는 함수"""
        img = np.array(data).reshape(self.height, self.width)  # 1D 배열을 2D 배열로 변환
        img[img == -1] = 255  # -1을 255로 변환
        img = img.astype(np.uint8)  # uint8로 변환
        return img

    def find_boundary(self, ori_image):
        """맵의 경계선을 찾는 함수"""
        image = ori_image.copy()
        image = cv2.inRange(image, 0, 0) | cv2.inRange(image, 255, 255)

        # 이미지의 경계선을 찾기 위해 Canny 엣지 검출기를 사용
        edges = cv2.Canny(image, threshold1=50, threshold2=150)

        # 경계선 확장을 위해 dilation을 사용
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        
        # 노이즈를 줄이기 위해 erosion을 사용
        binary_edges = cv2.erode(dilated_edges, None, iterations=1)

        # 연결된 구성 요소 분석
        num_labels, labels = cv2.connectedComponents(binary_edges)

        # 각 경계선의 좌표와 중심 좌표 저장
        centroids = []

        for label in range(1, num_labels):  # label 0은 배경
            points = np.where(labels == label)
            if len(points[0]) > 2: # 2개 이상만 
                # 경계선의 좌표
                boundary_points = list(zip(points[1], points[0]))
                
                # 중심 좌표 계산 (중심: x와 y 좌표의 평균)
                center_x = int(np.mean(points[1]))
                center_y = int(np.mean(points[0]))
                centroid = (center_x, center_y)

                centroids.append(centroid)


        return centroids

    def is_goal_safe(self, goal_x, goal_y, safety_radius=0.4, obstacle_value=100):
        """목표 지점 주변이 안전한지 확인하는 함수"""
        map_x = int((goal_x - self.origin.position.x) / self.resolution)
        map_y = int((goal_y - self.origin.position.y) / self.resolution)
        radius_pixels = int(safety_radius / self.resolution)

        # 장애물 범위 처리
        for y in range(map_y - radius_pixels, map_y + radius_pixels + 1):
            for x in range(map_x - radius_pixels, map_x + radius_pixels + 1):
                if x < 0 or y < 0 or x >= self.width or y >= self.height:
                    continue
                if self.data[y * self.width + x] == obstacle_value:
                    return False
        return True


    def finish_mapping(self):
        """맵핑 완료 시 초기 위치로 돌아가는 함수"""
        if self.goal_start:
            self.get_logger().info('Mapping already finished, returning to initial pose.')
            return  # 이미 목표 지점이 설정된 경우, 종료 처리

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

    # 초기 맵과 포즈가 수신될 때까지 기다림
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.is_init_pose and node.map:
            node.get_logger().info("Initial pose and map received. Starting...")
            break

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        rclpy.shutdown()