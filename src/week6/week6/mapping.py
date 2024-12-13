import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Twist, Quaternion, PoseStamped
from sensor_msgs.msg import LaserScan
import threading


class MapWithPose(Node):
    def __init__(self):
        super().__init__('mapping')
        self.scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/pose', self.pose_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
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

        pose = msg.pose
        # if pose가 초기 위치와 거의 같다면 mapping을 정지합니다.
        if self.is_init_pose and self.is_close(self.init_pose, pose):
            self.get_logger().info('Stop mapping')
            self.scan.destroy()
            self.pose_sub.destroy()
            self.vel_pub.destroy()
            self.destroy_node() # 노드 종료
    
    def is_close(self, pose1, pose2):
        '''
        두 위치가 거의 같은지 확인합니다.
        '''
        x1, y1 = pose1.position.x, pose1.position.y
        x2, y2 = pose2.position.x, pose2.position.y
        return (x1-x2)**2 + (y1-y2)**2 < 0.01

    def scan_callback(self, msg):
        '''
        회피 기동을 위한 레이저 스캔 데이터를 받아옵니다.
        1. 레이저 스캔 데이터를 받아옵니다.
        2. 앞 부분에 장애물이 있는지 확인합니다.
        3. 장애물이 있을 경우, 회피 기동을 수행합니다.
        4. 회피 기동을 수행합니다.
        '''
        # 1. 레이저 스캔 데이터를 받아옵니다.
        ranges = msg.ranges
        # 2. 앞 부분에 장애물이 있는지 확인합니다.
        front_ranges = ranges[:10] + ranges[-10:]
        min_range = min(front_ranges)
        if min_range < 1.0: # 1m 이내에 장애물이 있을 경우
            # 3. 장애물이 있을 경우, 회피 기동을 수행합니다.
            twist = self.avoid_obstacle()
        else:
            twist = Twist()
            twist.linear.x = 0.5
        # 4. 회피 기동을 수행합니다.
        self.vel_pub.publish(twist)
    
    def avoid_obstacle(self):
        '''
        장애물 회피 기동을 수행합니다.
        1. 장애물을 피해 회전합니다.
        2. 장애물을 피해 전진합니다.
        '''
        # 1. 장애물을 피해 회전합니다.
        twist = Twist()
        twist.angular.z = 0.5  # 좌회전 각속도 설정
        return twist
    

def main(args=None):
    rclpy.init()
    node = MapWithPose()
    while rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()


