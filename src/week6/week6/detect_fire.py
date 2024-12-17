import rclpy, os
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
import cv2,numpy as np 
from rclpy.qos import QoSProfile
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from nav_msgs.msg import OccupancyGrid
from PyQt5.QtCore import QThread, pyqtSignal
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from tf2_msgs.msg import TFMessage  
import sys, time
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration  # Duration을 가져옵니다.
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

odom_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,  # 퍼블리셔의 RELIABILITY와 동일하게 설정
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

BASELINK_TO_CAMERA = np.array([ 
    [0.000, 1.000, 0.000, 0.000],
    [-1.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 1.000, 0.244],
    [0.000, 0.000, 0.000, 1.000]])

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
ext_image = cv2.imread(os.path.join(dir_path, 'ext_orig.png'), cv2.IMREAD_GRAYSCALE)
man_image= cv2.imread(os.path.join(dir_path, 'man_orig.png'), cv2.IMREAD_GRAYSCALE)
EXT_IMG = {
    'image': ext_image,
    'size': (0.23,0.18), # meter
    'image_size': ext_image.shape,
    'type': 1
}
MAN_IMG = {
    'image': man_image,
    'size': (0.18,0.18) ,# meter
    'image_size': man_image.shape,
    'type': 2
}

class SIFTDetector():
    def __init__(self,ori_img,cap_img,types: int):
        self.ori_img = ori_img
        self.cap_img = cap_img
        self.result = False
        self.types = types
        self.detect()

    def detect(self):
        """
        SIFT 알고리즘을 사용하여 이미지 간 매칭 및 객체 탐지 수행 (강화된 검증 포함).
        """
        # SIFT 생성 및 특징점 추출
        self.sift = cv2.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ori_img, None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.cap_img, None)

        if self.des1 is None or self.des2 is None:
            print("Insufficient features in one of the images.")
            self.result = False
            return
        if len(self.kp1) < 2 or len(self.kp2) < 2:
            print("Insufficient keypoints in one of the images.")
            self.result = False
            return

        # 특징 매칭
        self.good_matches = self.match_features(self.des1, self.des2)

        # 충분한 매칭점이 있는지 확인
        if self.types == 1: # ext
            n = 35
        elif self.types == 2: # man
            n = 100
        if len(self.good_matches) > n: 
            try:
                # Homography 계산 및 검증
                self.homography, mask = self.compute_homography(self.good_matches)

                # 추가 검증: 투영 오류
                if not self.validate_projection_error(self.good_matches, mask):
                    self.result = False
                    return

                self.bounds = self.calculate_center_and_size(self.homography, self.ori_img)
                # 캡처 이미지의 좌표로 self.points 저장
                # 캡처 이미지의 좌표로 self.points 저장
                self.origin_points = [  
                    (self.kp1[match.queryIdx].pt[0], self.kp1[match.queryIdx].pt[1]) for match in self.good_matches
                ]
                self.real_points = [
                    (self.kp2[match.trainIdx].pt[0], self.kp2[match.trainIdx].pt[1]) for match in self.good_matches
                ]
                if len(self.bounds) < 2: # bounds가 없을 경우
                    self.result = False
                    return
                self.result = True
            except Exception as e:
                print(f"Error calculating homography: {e}")
                self.result = False
        else:
            print("Not enough good matches.")
            self.result = False

    def validate_projection_error(self, good_matches, mask, threshold=5.0):
        """
        투영 오류를 계산하여 매칭 품질을 추가적으로 검증.
        
        Args:
            good_matches (list): 매칭된 키포인트 리스트.
            mask (np.ndarray): RANSAC에 의해 필터링된 매칭 마스크.
            threshold (float): 허용 투영 오류 임계값.
        
        Returns:
            bool: 투영 오류가 허용 임계값 이하인지 여부.
        """
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), self.homography).reshape(-1, 2)

        errors = np.linalg.norm(projected_pts - dst_pts, axis=1)
        mean_error = np.mean(errors[mask.ravel() == 1])

        print(f"Mean projection error: {mean_error:.2f}")
        return mean_error < threshold

    def match_features(self,des1, des2,theshold=0.8):
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

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
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


class Selector():
    def __init__(self,origin_points,real_points: list, bounds: tuple):
        '''
        bounds: (center_x, center_y), (width, height)
        '''
        self.origin_points = origin_points
        self.real_points = real_points
        self.bounds = bounds
        self.select()


    def __call__(self):
        return self.origin_points_selected, self.real_points_selected
    def select(self):
        '''
        1.  임의로 30개 좌표를 선택합니다.
        '''
        indexs = np.random.choice(len(self.origin_points), 30, replace=False)
        self.real_points_selected = [self.real_points[i] for i in indexs]
        self.origin_points_selected = [self.origin_points[i] for i in indexs]



class Pointer:
    def __init__(self,camera_matrix: np.ndarray, dist_coeffs: np.ndarray,tf_map_camera: np.ndarray):
        self.K = camera_matrix
        self.D = dist_coeffs 
        self.tf_map_camera = tf_map_camera

    def __call__(self, pixel_list: list, matches_list: list,scale_real: tuple,scale_img: tuple):
        '''
        input: pixel_list: [(x1,y1),(x2,y2),...] # pixel 좌표계 ( 이미지 상에서의 좌표)

        matches_list: [(x1,y1),(x2,y2),...] # pixel 좌표계 (원본 이미지 상의 좌표)
        scale: (width, height) # meter 단위 이미지의 실제 크기

        output: (x,y) # meter 단위 (map 상에서의 좌표)
        '''
        self.pixel_list = pixel_list
        self.matches_list = matches_list
        self.scale_real = scale_real
        self.scale_img = scale_img
        self.pixel_to_scale()
        self.pnp_and_tf()
        self.final_tf()
        return self.pointer()
        
    def pixel_to_scale(self):
        '''
        1. 픽셀 좌표를 맵의 실제 크기(scale)를 기준으로 3D 좌표계로 변환합니다.
        2. (0,0)을 중심으로 변환합니다. 좌측 상단
        2. 변환된 3D 좌표 리스트를 반환합니다.
        '''
        self.scale_list = []
        for match in self.matches_list:
            x,y = match
            x = x/self.scale_img[0] * self.scale_real[0]
            y = y/self.scale_img[1] * self.scale_real[1]
            z = 0
            self.scale_list.append([x,y,z])

    def pnp_and_tf(self):
        '''
        1. 3D 포인트를 2D로 변환합니다.
        2. 변환된 2D를 반환합니다.
        이 함수를 통하여 tf를 구한다. 
        이 tf는 camera to image의 tf이다.
        '''
        object_points = np.array(self.scale_list) # 3d 데이터, meter 좌표계
        image_points = np.array(self.pixel_list) # 2d 데이터, pixel 좌표계

        # 이미 undistort된 상태이므로, undistortImage를 사용하지 않습니다.
        # 그러므로 K, D는 (3x3)의 단위 행렬입니다.
        self.K = np.eye(3) 
        self.D = np.zeros(5) # zero distortion coefficients
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.K, self.D
        )
        self.tf_camera_image =np.eye(4)
        self.tf_camera_image[:3, :3] = cv2.Rodrigues(rvec)[0]
        self.tf_camera_image[:3, 3] = tvec.flatten()
    
    def final_Stf(self):
        '''
        여기서 얻은 tf는 map to image의 tf이다.
        '''
        self.tf = np.dot(self.tf_map_camera, self.tf_camera_image)

    def pointer(self):
        '''
        변환 좌표계를 이용하여 이미지의 map상에서의 위치를 파악합니다.
        '''
        self.map_image_point = []
        for point in self.pixel_list:
            x,y = point
            point = np.array([x,y,0,1])
            point = np.dot(self.tf, point)
            self.map_image_point.append(point) # map 상에서의 좌표를 얻는다. --> (x,y,z) 좌표계로 변환된다.
        avg_x = np.mean([point[0] for point in self.map_image_point])
        avg_y = np.mean([point[1] for point in self.map_image_point])
        return avg_x, avg_y
    
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
        self.image_sub = self.create_subscription(CompressedImage, '/oakd/rgb/preview/image_raw/compressed', self.image_callback, img_qos)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.tf_map_odom_sub = self.create_subscription(TFMessage, '/tf', self.tf_map_odom_callback, 10)
        self.odom_baselink_sub = self.create_subscription(Odometry, '/odom', self.tf_odom_base_link_callback, odom_profile)
        self.odom_to_base_link = None
        self.initing()
        self.image_load()
    
    def tf_map_odom_callback(self, msg):
        '''
        '''
        map_to_odom = None 
        for transform in msg.transforms:
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            map_to_odom= np.array([
                [rotation.x, rotation.y, rotation.z, translation.x],
                [rotation.y, rotation.x, rotation.z, translation.y],
                [rotation.z, rotation.z, rotation.x, translation.z],
                [0, 0, 0, 1]
            ])
        if map_to_odom is None:
            self.get_logger().warn('Failed to get map to odom transform.')
            return
        self.tf_map_odom = map_to_odom

    def tf_odom_base_link_callback(self, msg):
        '''
        '''
        translation = msg.pose.pose.position
        rotation = msg.pose.pose.orientation
        self.odom_to_base_link = np.array([
            [rotation.x, rotation.y, rotation.z, translation.x],
            [rotation.y, rotation.x, rotation.z, translation.y],
            [rotation.z, rotation.z, rotation.x, translation.z],
            [0, 0, 0, 1]
        ])
        if self.odom_to_base_link is None:
            self.get_logger().warn('Failed to get odom to base_link transform.')
            return
        if self.tf_map_odom is None or self.odom_to_base_link is None:
            self.get_logger().warn('Failed to get map to odom transform.')
            return
        if self.tf_map_base_link is not None and self.tf_map_odom is not None:
            self.tf_map_base_link = np.dot(self.tf_map_odom, self.odom_to_base_link)
            self.tf_map_camera = np.dot(self.tf_map_base_link, BASELINK_TO_CAMERA) # map to camera tf
            self.get_logger().info('TF obtained successfully')
    

    def map_callback(self, msg):
        """맵 데이터를 처리하고 목표 지점을 설정하는 콜백"""
        if msg is None or len(msg.data) == 0:
            self.get_logger().warn('Received empty map or no map data.')
            return
        # 맵 정보 처리
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.data = msg.data
        self.map_img = self.data_to_image(self.data)  # 맵 데이터를 이미지로 변환

    def data_to_image(self, data):
        '''
        data를 이미지로 변환합니다.
        0은 255로 변환합니다.
        -1은 0으로 변환합니다.
        흑백 이미지를 컬러 이미지로 변환합니다.
        '''
        data = np.array(data).reshape((self.height, self.width))
        data = np.where(data == 0, 255, data)  # -1은 미지의 값으로 255로 변환합니다.
        data = np.where(data == -1,0, data) # -1은 미지의 값으로 0으로 변환합니다.
        data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_GRAY2BGR) # 흑백 이미지를 컬러 이미지로 변환합니다.
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # OpenCV 이미지를 Qt 이미지로 변환합니다. 
        return data

    def initing(self):
        self.types = 0

        # image
        self.image_height = None
        self.image_width = None
        self.cam_image = None
        self.undistort_image = None

        self.bounds = None
        
        self.tf_map_camera_msg = None
        self.tf_map_camera = None # map to camera tf
        # info
        self.K = None
        self.D = None

        # map
        self.width = 10
        self.height = 10
        self.map_img = None

        self.point_img_map = None

        # pose
        self.pose = None

        self.tf_map_base_link = None
        self.tf_map_odom = None
        self.tf_map_camera

    def image_load(self):
        # 이미지 
        self.man_img = MAN_IMG['image']
        self.ext_img = EXT_IMG['image']

    def image_callback(self, msg):
        if msg is None:
            self.get_logger().warn('No image received.')
            return 
        
        # 메시지를 NumPy 배열로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)  # 바이너리 데이터를 NumPy 배열로 변환
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV 형식으로 디코딩

        if image is None:
            self.get_logger().error('Failed to decode image.')
            return

        # 이미지 처리: 회색조로 변환
        self.cam_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 이미지 왜곡 보정 (예시로 호출)
        self.undistort_image = self.change_image()

        # 특정 처리 수행 (예시로 호출)
        self.detect_and_pointing()


    def detect_and_pointing(self):
        '''
        1. 이미지를 받아 와서 type을 확인합니다. 
        '''
        self.types,self.origin_points,self.real_points, self.bounds = self.check()
        if self.tf_map_camera is None:
            self.get_logger().info('No TF Map Camera')
            return
        pointer = Pointer(self.K, self.D, self.tf_map_camera) # pointer를 생성합니다.
        if self.types == EXT_IMG['type']:
            self.get_logger().info('Ext Image Detected')
        elif self.types == MAN_IMG['type']:
            self.get_logger().info('Man Image Detected')
        elif self.types == 0:
            self.get_logger().info('No Image Detected')
            return 
        
        self.get_logger().info(f'Bounds: {self.bounds}')
        selector = Selector(self.origin_points,self.real_points, self.bounds) # selector를 생성합니다.
        origin_points, real_points = selector() # 선택된 좌표를 받아옵니다.
        size = EXT_IMG['size'] if self.types == EXT_IMG['type'] else MAN_IMG['size'] if self.types==MAN_IMG['type'] else None
        if size is None:
            return
        image_piexl_scale = EXT_IMG['image_size'] if self.types == EXT_IMG['type'] else MAN_IMG['image_size'] 

        result = pointer(origin_points, real_points, size, image_piexl_scale)
        # pointer를 호출합니다.
        self.point_img_map = result # map 상에서의 좌표를 저장합니다. (x,y) 좌표계로 저장됩니다.
        self.get_logger().info(f'Pointer: {result}')

    
    def check(self):
        result = SIFTDetector(self.ext_img,self.undistort_image,types=EXT_IMG['type'])
        if result.result:
            return EXT_IMG['type'], result.origin_points,result.real_points, result.bounds
        result = SIFTDetector(self.man_img,self.undistort_image,types=MAN_IMG['type'])
        if result.result:
            return MAN_IMG['type'], result.origin_points,result.real_points, result.bounds
        return 0, 0, 0 ,0 # 매칭된 이미지가 없을 경우 0을 반환합니다.
        
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
        d: (8)
        k: (9)
        '''
        if msg is None:
            self.get_logger().info('No Camera Info')
            return
        
        self.K  = np.array(msg.k).reshape((3,3))
        self.D = np.array(msg.d[:5]).reshape((1,5))
        self.image_height = msg.height
        self.image_width = msg.width
        self.get_logger().info('Camera Info Received')

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
        if self.cam_image is None or self.K is None or self.D is None:
            return
        undistorted_image = cv2.undistort(self.cam_image, self.K, self.D)
        return undistorted_image



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
        self.image_size = (500, 500)

        # 메인 위젯과 레이아웃 설정
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        self.window.setCentralWidget(self.central_widget)

        # 4개의 창 생성
        self.create_views()

        # 초기화
        self.initing()

        # 원본 이미지 설정
        self.origin_image1 = self.node.man_img
        self.origin_image2 = self.node.ext_img
    
    def initing(self):
        '''
        '''
        empty_img = np.ones((500, 500), dtype=np.uint8)*255
        self.map_label.setPixmap(self.convert_to_pixmap(empty_img, self.image_size))
        self.camera_label.setPixmap(self.convert_to_pixmap(empty_img, self.image_size))
        self.detect_label.setText('Detected: None')
        self.coord_label.setText('Image point is None')
        
        self.point_pixel = None
        self.ext_img_point = None
        self.man_img_point = None  

    def create_views(self):
        """
        4개의 이미지를 표시할 창을 생성.
        """
        # 2x2 레이아웃 설정
        layout = QGridLayout()
        self.main_layout.addLayout(layout)
        image_size = self.image_size

        # 왼쪽 위: 맵 이미지
        self.map_label = QLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setFixedSize(image_size[0], image_size[1])  # 크기 고정
        layout.addWidget(self.map_label, 0, 0)  # (row 0, column 0)

        # 왼쪽 아래: 카메라 이미지
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(image_size[0], image_size[1])
        layout.addWidget(self.camera_label, 1, 0)  # (row 1, column 0)

        # 오른쪽 위: 탐지 여부
        self.detect_label = QLabel()
        self.detect_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detect_label, 0, 1)  # (row 0, column 1)

        # 오른쪽 아래: 좌표
        self.coord_label = QLabel()
        self.coord_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.coord_label, 1, 1)  # (row 1, column 1)

        # 저장 버튼
        save_button = QPushButton('Save Map Image')
        save_button.clicked.connect(self.map_saving)
        self.main_layout.addWidget(save_button)



    def convert_to_pixmap(self, image, target_size=None):
        """
        numpy 이미지를 QPixmap으로 변환.

        Args:
            image (numpy.ndarray): OpenCV 이미지.
            target_size (tuple, optional): (width, height)로 리사이즈할 크기. 기본값은 None.

        Returns:
            QPixmap: QPixmap 객체.
        """
        if image is None:
            # 빈 이미지를 반환
            empty_image = np.ones((500, 500), dtype=np.uint8)*255
            image = empty_image

        # Grayscale 이미지 처리
        if image.ndim == 2:
            h, w = image.shape
            qt_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        # RGB 이미지 처리
        elif image.ndim == 3:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            raise ValueError("Unsupported image format")

        # QImage → QPixmap 변환
        pixmap = QPixmap.fromImage(qt_image)

        # target_size로 크기 변경
        if target_size:
            pixmap = pixmap.scaled(target_size[0], target_size[1], Qt.KeepAspectRatio)

        return pixmap

    def update_gui(self):
        """
        탐지 결과를 업데이트.
        """
        result = "Man" if self.node.types == MAN_IMG['type'] else "Ext" if self.node.types == EXT_IMG['type'] else "None"
        self.detect_label.setText(f'Detected: {result}')

        if self.node.point_img_map is None:
            self.coord_label.setText("Image point is None")
            point_pixel = None
        else:
            point_pixel = self.point_pixeling(self.node.point_img_map[0], self.node.point_img_map[1])
            string = f'If map origin is (0,0)Image Point: {self.node.point_img_map}\n'
            string += f'Pixel Point: {point_pixel}'
            self.coord_label.setText(string)
        
        if self.node.types == MAN_IMG['type']:
            self.man_img_point = point_pixel
        elif self.node.types == EXT_IMG['type']:
            self.ext_img_point = point_pixel
        if self.node.map_img is not None and self.node.cam_image is not None:
            map_pixmap = self.convert_to_pixmap(self.node.map_img, self.image_size)
            camera_pixmap = self.convert_to_pixmap(self.node.cam_image, self.image_size)
            self.map_label.setPixmap(map_pixmap)
            self.camera_label.setPixmap(camera_pixmap)

        if self.node.map_img is None:
            map_img = np.zeros((self.node.height, self.node.width), dtype=np.uint8) # 맵이 없을 경우 빈 이미지를 생성합니다.
            map_pixmap = self.convert_to_pixmap(map_img, self.image_size)          
            self.map_label.setPixmap(map_pixmap)
        if self.node.cam_image is None:
            camera_img = np.zeros((self.node.height, self.node.width), dtype=np.uint8)
            camera_pixmap = self.convert_to_pixmap(camera_img, self.image_size)
            self.camera_label.setPixmap(camera_pixmap)
        if self.node.bounds is not None and self.node.bounds != 0:
            self.draw_box(self.node.cam_image, self.node.bounds)
        self.draw_points(self.node.map_img)
    
    def draw_points(self, map_image):
        '''
        map_pixmap에 점을 그립니다.
        색은 빨간색과 파란색으로 나뉩니다.
        man: 파란색, ext: 빨간색

        '''
        blue = (255,0,0)
        red = (0,0,255)
        green = (0,255,0)
        pose = self.node.pose
        if map_image is None:
            return 
        image = map_image.copy() # 복사본을 만듭니다.
        if pose is not None:
            x_pixel, y_pixel = self.point_pixeling(pose[0], pose[1])
            cv2.circle(image, (x_pixel, y_pixel), 2, green, -1) # 크기: 2
        if self.ext_img_point is not None:
            cv2.circle(image, self.ext_img_point, 2, red, -1) # 크기: 
        if self.man_img_point is not None:
            cv2.circle(image, self.man_img_point, 2, blue, -1)
        map_pixmap = self.convert_to_pixmap(image, self.image_size)
        self.map_label.setPixmap(map_pixmap)
        self.pointed_image = image
    
    def map_saving(self):
        cv2.imwrite('map_image.png', self.pointed_image)
    
    def draw_box(self,image, bounds):
        '''
        bounds를 이용하여 box를 그립니다. (x,y,w,h)
        '''
        center, size = bounds
        x = int(center[0])
        y = int(center[1])
        w = int(size[0])
        h = int(size[1])
        cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
        self.camera_label.setPixmap(self.convert_to_pixmap(image, self.image_size))

    def point_pixeling(self, x, y):
        '''
        좌표를 받아와서 변환 후 pixel 좌표로 출력합니다. 
        '''
        if self.node.map_img is None or self.node.undistort_image is None:
            return
        if self.node.width is None or self.node.height is None or self.node.origin is None and self.node.resolution is None:
            return
        x_pixel = int((x - self.node.origin.position.x) / self.node.resolution)
        y_pixel = int((y - self.node.origin.position.y) / self.node.resolution)
        return x_pixel, y_pixel

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
    ros_thread.update_signal.connect(gui.update_gui)
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