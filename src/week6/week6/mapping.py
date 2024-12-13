import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Twist, Quaternion, PoseStamped
from sensor_msgs.msg import LaserScan
import threading
import cv2, numpy as np


class MapWithPose(Node):
    def __init__(self):
        super().__init__('mapping')
        # self.scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/pose', self.pose_callback, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.is_init_pose = False

    def pose_callback(self, msg):
        '''
        위치를 주기적으로 받습니다. 
        1. 초기 위치를 받아오게 합니다. 
        '''
        if not self.is_init_pose:
            self.get_logger().info('Get initial pose')
            self.init_pose = msg.pose
            self.is_init_pose = True
            return

        self.pose = msg.pose
        # if pose가 초기 위치와 거의 같다면 mapping을 정지합니다.
        if self.is_init_pose and self.is_close():
            self.get_logger().info('Stop mapping')
            self.scan.destroy()
            self.pose_sub.destroy()
            self.vel_pub.destroy()
            self.destroy_node() # 노드 종료
        else:
            self.get_logger().info('Continue mapping')
    
    def is_close(self):
        '''
        두 위치가 거의 같은지 확인합니다.
        '''
        x1, y1 = self.init_pose.position.x, self.init_pose.position.y
        x2, y2 = self.pose.position.x, self.pose.position.y
        return (x1-x2)**2 + (y1-y2)**2 < 0.01
    
    def map_callback(self, msg):
        '''
        맵을 받아옵니다.
        1. 맵을 받아옵니다.
        2. 맵에서 -1, 0 100을 각각 255, 0, 100으로 바꿔 이미지로 출력합니다.
        3. 맵에서 -1과 0의 경계선을 찾아 출력합니다
        4. 경계선을 찾아서 출력합니다.
        '''
        self.map = msg
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.data = msg.data
        self.get_logger().info('Get map')
        # 2. 맵에서 -1, 0 100을 각각 255, 0, 50으로 바꿔 이미지로 출력합니다.
        map_img = self.data_to_image(self.data)
        if np.any(map_img == 255) is False:
            self.get_logger().info('Mapping is done')
            self.finish_mapping()
        points = self.find_boundary(map_img)
        goal_point = self.find_goal(points)
        self.get_logger().info(f'Goal Point: {goal_point}')
        goal_pose = PoseStamped()
        goal_pose.pose.position.x = goal_point[0]
        goal_pose.pose.position.y = goal_point[1]
        self.goal_pose_pub.publish(goal_pose)

    def find_goal(self, points):
        '''
        현재의 포즈와 points를 이용하여 goal point를 찾습니다.
        '''
        min_distance = 100
        for point in points:
            if self.distance(point) < min_distance:
                goal_point = point
                min_distance = self.distance(point)
        return goal_point
    
    def distance(self,point):
        x = self.pose.position.x
        y = self.pose.position.y
        theta = self.pose.orientation.z
        x1, y1 = point
        return np.sqrt((x-x1)**2 + (y-y1)**2)
    def data_to_image(self, data):
        '''
        맵 데이터를 이미지로 변환합니다.
        '''
        img = np.array(data).reshape(self.height, self.width)
        img = np.where(img==-1, 255, img)
        img = np.where(img==0, 0, img)
        img = np.where(img==100, 50, img)
        img = img.astype(np.uint8)
        return img
    
    def find_boundary(self, image):
        '''
        맵에서 경계선을 찾습니다.
        여기서 픽셀 값이 급격하게 바뀌는 구간 (0 to 255)
        알고리즘을 찾아 경계선을 찾아 출력합니다.
        '''
        # 3. 맵에서 -1과 0의 경계선을 찾아 출력합니다
        boundary = cv2.Canny(image, 0, 255)
        return boundary
    
    def finish_mapping(self):
        '''
        맵핑이 끝났을 때 수행할 작업을 수행합니다.
        '''
        self.get_logger().info('Finish mapping')
        self.map.destroy()
        self.pose_sub.destroy()
        self.goal_pose_pub.destroy()
        self.destroy_node()\
        
def main(args=None):
    rclpy.init()
    node = MapWithPose()
    while rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()
    

