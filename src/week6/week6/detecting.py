import rclpy    
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
import cv2
from rclpy.qos import QoSProfile
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
# QoS 프로파일 설정
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,  # 신뢰성: 데이터 보장(RELIABLE) 또는 최선형(BEST_EFFORT)
    history=10,                              # 히스토리 깊이
    durability=DurabilityPolicy.VOLATILE    # 지속성: VOLATILE (구독 중일 때만 데이터 유지)
)
ext_img = './src/week6/week6/image1.png'
main_img= './src/week6/week6/image2.png'
class DetectImage(Node):
    def __init__(self):
        '''
        이미지를 받아 맵에 표시하는 노드를 생성합니다.
        '''
        super().__init__('detecting')
        self.info_sub = self.create_subscription(CameraInfo, '/oakd/rgb/preview/camera_info', self.info_callback, 10)
        self.image_sub = self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.main_img = cv2.imread()
    def image_callback(self, msg):
        data = msg.data
        bridge = CvBridge()
        image1 = bridge.imgmsg_to_cv2(data, 'bgr8')
        self.get_logger().info('image get')
        is_detect = self.is_detect(image1)
        if is_detect:
            self.get_logger().info('Detected')
            self.ch_image(image1)

    def is_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresholded)
        total_pixels = image.shape[0] * image.shape[1]
        white_ratio = white_pixels / total_pixels
        return white_ratio > 0.5


    def info_callback(self, msg):
        '''
        예시 
        header:
            stamp:
                sec: 1734051505
                nanosec: 363371767
            frame_id: oakd_rgb_camera_optical_frame
        height: 250
        width: 250
        distortion_model: rational_polynomial
        d:
        - -3.4751670360565186
        - -38.5734748840332
        - 0.00034309603506699204
        - -9.377215610584244e-05
        - 286.4400939941406
        - -3.6408045291900635
        - -36.68898010253906
        - 279.0523681640625
        k:
        - 202.61964416503906
        - 0.0
        - 124.34600067138672
        - 0.0
        - 202.61964416503906
        - 127.28642272949219
        - 0.0
        - 0.0
        - 1.0
        r:
        - 1.0
        - 0.0
        - 0.0
        - 0.0
        - 1.0
        - 0.0
        - 0.0
        - 0.0
        - 1.0
        p:
        - 202.619 MapWithPose()64416503906
        - 0.0
        - 124.34600067138672
        - 0.0
        - 0.0
        - 202.61964416503906
        - 127.28642272949219
        - 0.0
        - 0.0
        - 0.0
        - 1.0
        - 0.0
        binning_x: 0
        binning_y: 0
        roi:
        x_offset: 0
        y_offset: 0
        height: 0
        width: 0
        do_rectify: false
        ---
        '''
        if msg is None:
            self.get_logger().info('No Camera Info')
            return
        self.width = msg.width
        self.height = msg.height
        self.k = msg.k
        self.d = msg.d
        self.r = msg.r
        self.p = msg.p
        self.binning_x = msg.binning_x
        self.binning_y = msg.binning_y
        self.roi = msg.roi

    def ch_image(self, image):
        '''
        1. 이미지를 받아옵니다.
        2. info를 이용하여 undistort를    

def main(args=None):
    rclpy.init()
    node = MapWithPose()
    while rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown() 수행합니다.
        3. undistort된 이미지를 출력합니다.
        '''

def main(args=None):
    rclpy.init()
    node = DetectImage()
    while rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()
