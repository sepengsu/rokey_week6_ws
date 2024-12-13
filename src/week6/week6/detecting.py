import rclpy    
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
import cv2,numpy as np 
from rclpy.qos import QoSProfile
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, QoSHistoryPolicy

qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,      # 신뢰성: 데이터 보장(RELIABLE) 또는 최선형(BEST_EFFORT)
    history=QoSHistoryPolicy.KEEP_LAST,          # 히스토리 정책
    depth=10,                                    # KEEP_LAST일 경우 히스토리 깊이
    durability=DurabilityPolicy.VOLATILE         # 지속성: VOLATILE (구독 중일 때만 데이터 유지)
)

ext_img = './src/week6/week6/image1.png'
main_img= './src/week6/week6/image2.png'
EXT = 1
MAIN = 2
class DetectImage(Node):
    def __init__(self):
        '''
        이미지를 받아 맵에 표시하는 노드를 생성합니다.
        '''
        super().__init__('detecting')
        self.info_sub = self.create_subscription(CameraInfo, '/oakd/rgb/preview/camera_info', self.info_callback, 10)
        self.image_sub = self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.main_img = cv2.imread(main_img, cv2.IMREAD_GRAYSCALE) # 회색 이미지로 읽어옵니다.
        self.ext_img = cv2.imread(ext_img, cv2.IMREAD_GRAYSCALE) # 회색 이미지로 읽어옵니다.

    def image_callback(self, msg):
        if msg is None:
            self.get_logger().warn('no image acc')
            return 
        data = msg.data
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")

        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 회색 이미지로 변환합니다.
        self.ch_image()
        self.detect()

    def detect(self):
        '''
        1. 이미지를 받아옵니다.
        2. 이미지를 처리합니다.
        3. 이미지를 출력합니다.
        '''
        result = self.check()
        if result == EXT:
            self.get_logger().info('Ext Image Detected')
        elif result == MAIN:
            self.get_logger().info('Main Image Detected')
        else:
            self.get_logger().info('No Image Detected')
    
    def check(self):
        '''
        1. ext1과 img를 비교합니다.
        방법론 
        1. SIFT를 사용하여 특징점을 찾습니다.
        2. 특징점을 매칭합니다.
        3. 특징점의 개수를 확인합니다.
        4. 특징 매칭 결과과 15개 이상이면 True, 아니면 False를 반환합니다.
        '''
        # SIFT 알고리즘 초기화
        sift = cv2.SIFT_create()

        # 특징점과 디스크립터 추출
    
        keypoints1, descriptors1 = sift.detectAndCompute(self.ext_img, None)
        keypoints2, descriptors2 = sift.detectAndCompute(self.main_img, None)
        keypoints3, descriptors3 = sift.detectAndCompute(self.image, None)


        # 특징 매칭을 위한 BFMatcher 초기화
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # 매칭 수행
        matches = bf.match(descriptors1, descriptors3) # ext1과 img 매칭
        if len(matches) > 15:
            return EXT
        
        matches = bf.match(descriptors2, descriptors3) # main과 img 매칭
        if len(matches) > 15:
            return MAIN
        return 0
    
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

    def ch_image(self):
        '''
        1. 이미지를 받아옵니다.
        2. info를 이용하여 undistort를 진행합니다. 
        3. 이미지를 반환합니다. 
        '''
        K = np.array(self.k).reshape((3,3))
        D = np.array(self.d).reshape((1, 8))
        self.image = cv2.undistort(self.image, K, D)

def main(args=None):
    rclpy.init()
    node = DetectImage()
    while rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()
