import rclpy    
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
import cv2,numpy as np 
from rclpy.qos import QoSProfile
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
info_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)
img_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,  # 안정적인 전송
    durability=QoSDurabilityPolicy.VOLATILE,   # 이전 메시지 저장 안 함
    history=QoSHistoryPolicy.KEEP_LAST,    # 최신 메시지만 유지
    depth=10                                   # 최대 10개의 메시지 버퍼
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
        self.info_sub = self.create_subscription(CameraInfo, '/oakd/rgb/preview/camera_info', self.info_callback, info_qos)
        self.image_sub = self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.image_callback, img_qos)
        self.main_img = cv2.imread(main_img, cv2.IMREAD_GRAYSCALE) # 회색 이미지로 읽어옵니다.
        self.ext_img = cv2.imread(ext_img, cv2.IMREAD_GRAYSCALE) # 회색 이미지로 읽어옵니다.

    def image_callback(self, msg):
        if msg is None:
            self.get_logger().warn('no image acc')
            return 
        data = msg.data
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        self.get_logger().info('이미지 ok')
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 회색 이미지로 변환합니다.
        self.change_image()
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
        header:
            stamp:
                sec: 1734080480
                nanosec: 101003283
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
        - 202.61964416503906
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

    def change_image(self):
        '''
        1. 이미지를 받아옵니다.
        2. info를 이용하여 undistort를 진행합니다. 
        3. 이미지를 반환합니다. 

        데이터 포멧 
        header:
            stamp:
                sec: 1734080303
                nanosec: 866855742
            frame_id: oakd_rgb_camera_optical_frame
        height: 250
        width: 250
        encoding: bgr8
        is_bigendian: 0
        step: 750
        data:  -- array 이미지 
        '''
        K = np.array(self.k).reshape((3,3))
        D = self.d[:5]
        D = np.array(D).reshape((1,5))  # 1x5
        # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (self.width, self.height), 1, (self.width, self.height))
        undistorted_image = cv2.undistort(self.image, K, D)
        self.image = undistorted_image

def main(args=None):
    rclpy.init()
    node = DetectImage()
    while rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()





'''

transforms:
- header:
    stamp:
      sec: 1734086969
      nanosec: 681052987
    frame_id: map
  child_frame_id: odom
  transform:
    translation:
      x: 20.16926142645884
      y: 11.039336521688346
      z: -0.12570523118490198
    rotation:
      x: -0.008888777910641361
      y: 0.01704839010350562
      z: 0.9995527887709227
      w: 0.0229033727299089

'''