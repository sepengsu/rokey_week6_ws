import rclpy, os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
import cv2,numpy as np 
from rclpy.qos import QoSProfile
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from PyQt5.QtCore import QThread, pyqtSignal
import sys

import tf2_ros # tf2_ros는 tf2를 사용하기 위한 패키지입니다.
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
dir_path = get_package_share_directory('week6')
ext_img = os.path.join(dir_path, 'ext_orig.png')
man_img= os.path.join(dir_path, 'man_orig.png')
EXT = 1
MAIN = 2

class SIFTDetector():
    def __init__(self,ori_img,cap_img,types:int):
        self.ori_img = ori_img
        self.cap_img = cap_img
        self.types = types
        
        self.detect()

    def detect(self):
        """
        SIFT 알고리즘을 사용하여 이미지 간 매칭 및 객체 탐지 수행.
        """
        # SIFT 생성 및 특징점 추출
        self.sift = cv2.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ori_img, None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.cap_img, None)

        if self.des1 is None or self.des2 is None:
            print("Insufficient features in one of the images.")
            return

        # 특징 매칭
        self.good_matches = self.match_features(self.des1, self.des2)

        # 충분한 매칭점이 있는지 확인
        if len(self.good_matches) > 15:
            # Homography 계산
            try:
                self.homography, _ = self.compute_homography(self.good_matches)
                self.bounds = self.calculate_center_and_size(self.homography, self.ori_img)
                self.result = self.types
            except Exception as e:
                print(f"Error calculating homography: {e}")
                self.result = 0
        else:
            print("Not enough good matches.")
            self.result = 0


    def match_features(self,des1, des2,theshold=0.7):
        """
        FLANN 기반 매칭을 수행하고 좋은 매칭점을 반환합니다.
        """
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test 적용
        good_matches = []
        for m, n in matches:
            if m.distance < theshold * n.distance:
                good_matches.append(m)
        return good_matches
    
    def compute_homography(self, good_matches):
        """
        Homography를 계산하여 변환 행렬을 반환합니다.
        """
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None:
            raise ValueError("Homography calculation failed.")
        return homography, mask

    def calculate_center_and_size(self, homography, template_img):
        """
        Homography를 이용하여 캡처된 이미지 상에서 중심 좌표와 크기를 계산합니다.

        Args:
            homography (np.ndarray): 템플릿 이미지와 캡처된 이미지 간의 변환 행렬.
            template_img (np.ndarray): 템플릿 이미지 (원본).
        
        Returns:
            tuple: 중심 좌표 (center_x, center_y)와 크기 (width, height).
        """
        # 템플릿 이미지의 경계 좌표 정의
        h, w = template_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Homography를 이용해 템플릿 이미지 경계를 캡처된 이미지 좌표계로 변환
        dst = cv2.perspectiveTransform(pts, homography)

        # 캡처된 이미지의 좌표계에서 경계 추출
        x_min = np.min(dst[:, 0, 0])
        y_min = np.min(dst[:, 0, 1])
        x_max = np.max(dst[:, 0, 0])
        y_max = np.max(dst[:, 0, 1])

        # 중심 좌표와 크기 계산
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return (center_x, center_y), (width, height)

class ROSNodeThread(QThread):
    """
    ROS 노드를 별도의 QThread에서 실행.
    """
    update_signal = pyqtSignal()  # GUI 업데이트를 위한 신호

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.running = True

    def run(self):
        """
        ROS 노드 실행 루프.
        """
        while self.running:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            self.update_signal.emit()  # GUI 업데이트 신호

    def stop(self):
        """
        스레드를 종료합니다.
        """
        self.running = False
        self.wait()

class DetectImage(Node):
    def __init__(self):
        '''
        이미지를 받아 맵에 표시하는 노드를 생성합니다.
        '''
        super().__init__('detecting')
        self.get_logger().info('Detecting Node Started')
        self.info_sub = self.create_subscription(CameraInfo, '/oakd/rgb/preview/camera_info', self.info_callback, info_qos)
        self.image_sub = self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.image_callback, img_qos)

        self.result = 0
        self.image_load()

        self.tf_transform_get()
    
    def tf_transform_get(self):
        self.tf_buffer = Buffer() # tf2_ros.Buffer()를 사용하여 tf2_ros.Buffer를 초기화합니다.
        self.tf_listener = TransformListener(self.tf_buffer, self) # tf2_ros.TransformListener를 사용하여 tf2_ros.TransformListener를 초기화합니다.
        self.get_logger().info('TF2 Ready')
        from_frame = 'map'
        to_frame = 'oakd_rgb_camera_frame'
        # data = self.get_transform(from_frame, to_frame)
        
    
    def image_load(self):
        # 이미지 
        self.man_img = cv2.imread(man_img, cv2.IMREAD_GRAYSCALE) # 회색 이미지로 읽어옵니다.
        if self.man_img is None:
            self.get_logger().warn('No Main Image')
            raise ValueError('No Main Image')
        self.ext_img = cv2.imread(ext_img, cv2.IMREAD_GRAYSCALE) # 회색 이미지로 읽어옵니다.
        if self.ext_img is None:
            self.get_logger().warn('No Ext Image')
            raise ValueError('No Ext Image')
        

    def image_callback(self, msg):
        if msg is None:
            self.get_logger().warn('no image acc')
            return 
        data = msg.data
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 회색 이미지로 변환합니다.
        self.change_image()
        self.detect()

    def detect(self):
        '''
        1. 이미지를 받아옵니다.
        2. 이미지를 처리합니다.
        3. 이미지를 출력합니다.
        '''
        self.result = self.check()
        if self.result == EXT:
            self.get_logger().info('Ext Image Detected')
        elif self.result == MAIN:
            self.get_logger().info('Man Image Detected')
        else:
            self.get_logger().info('No Image Detected')
    
    def check(self):
        '''
        '''
        result = SIFTDetector(self.ext_img,self.image,EXT)
        if result.result == EXT:
            # ext1과 img가 일치하는 경우

            return EXT
    def pixel_to_scale(self,pixel,scale):
        pass

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
        undistorted_image = cv2.undistort(self.image, K, D)
        self.image = undistorted_image



class GUI:
    def __init__(self, node):
        """
        GUI 클래스 초기화.

        Args:
            node (DetectImage): 탐지 노드 객체.
        """
        self.node = node
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle('Image Viewer')
        self.window.setGeometry(100, 100, 1200, 800)

        # 메인 위젯과 레이아웃 설정
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        self.window.setCentralWidget(self.central_widget)

        # 4개의 창 생성
        self.create_image_views()

        # 원본 이미지 설정
        self.origin_image1 = self.node.man_img
        self.origin_image2 = self.node.ext_img

    def create_image_views(self):
        """
        4개의 이미지를 표시할 창을 생성.
        """
        for i in range(4):
            layout = QHBoxLayout()

            # 왼쪽: 원본 이미지
            original_label = QLabel()
            original_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(original_label)

            # 오른쪽: 탐지된 이미지
            detected_label = QLabel()
            detected_label.setAlignment(Qt.AlignCenter)
            detected_label.setText(f"Detected Image {i + 1}")
            layout.addWidget(detected_label)

            # 레이아웃 추가
            self.main_layout.addLayout(layout)

    def update_images(self, detected_images):
        """
        원본 이미지와 탐지 이미지를 업데이트.

        Args:
            detected_images (list): 탐지된 이미지 리스트 (numpy.ndarray 형식).
        """
        original_images = [self.origin_image1, self.origin_image2]

        for i, (original_img, detected_img) in enumerate(zip(original_images, detected_images)):
            if i >= self.main_layout.count():
                break

            layout = self.main_layout.itemAt(i)
            if layout is None:
                continue

            original_label = layout.itemAt(0).widget()
            detected_label = layout.itemAt(1).widget()

            # 원본 이미지 업데이트
            if original_img is not None:
                original_pixmap = self.convert_to_pixmap(original_img)
                original_label.setPixmap(original_pixmap)

            # 탐지 이미지 업데이트
            if detected_img is not None:
                detected_pixmap = self.convert_to_pixmap(detected_img)
                detected_label.setPixmap(detected_pixmap)

    def convert_to_pixmap(self, image):
        """
        numpy 이미지를 QPixmap으로 변환.

        Args:
            image (numpy.ndarray): OpenCV 이미지.

        Returns:
            QPixmap: QPixmap 객체.
        """
        if image.ndim == 2:  # Grayscale 이미지 처리
            h, w = image.shape
            qt_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:  # RGB 이미지 처리
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(qt_image)

    def update_detect(self):
        """
        탐지 결과를 업데이트.
        """
        if self.node.image is None:
            return
        black_image = np.zeros_like(self.node.image)
        if self.node.result == EXT:
            detected_images = [self.node.image, black_image]
        elif self.node.result == MAIN:
            detected_images = [black_image, self.node.image]
        else:
            detected_images = [black_image, black_image]
        self.update_images(detected_images)

    def show(self):
        """
        GUI를 실행.
        """
        self.window.show()
        sys.exit(self.app.exec_())

def main():
    """
    메인 함수로 ROS2 노드와 PyQt5 GUI를 동시에 실행.
    """
    rclpy.init()

    # DetectImage 노드 생성
    node = DetectImage()

    # PyQt5 기반 GUI 생성
    gui = GUI(node)

    # ROSNodeThread를 사용하여 ROS2 노드 실행
    ros_thread = ROSNodeThread(node)
    ros_thread.update_signal.connect(gui.update_detect)  # 탐지 결과를 GUI에 전달
    ros_thread.start()

    # PyQt5 GUI 실행
    try:
        gui.show()
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 ROS2와 스레드 정리
        ros_thread.stop()
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