import cv2
import numpy as np
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)


def detect_largest_face(image):
    # Chuyển ảnh sang định dạng phù hợp (BGR -> RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện tất cả khuôn mặt
    faces = face_app.get(img_rgb)
    
    if len(faces) == 0:
        return None  # Không phát hiện khuôn mặt nào
    
    # Tính toán diện tích của các khuôn mặt
    largest_face = None
    largest_area = 0
    
    for face in faces:
        # Lấy bounding box của khuôn mặt
        bbox = face.bbox  # Lưu ý là [x1, y1, x2, y2]
        
        # Tính diện tích của bounding box
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = w * h
        
        # Chọn khuôn mặt có diện tích lớn nhất
        if area > largest_area:
            largest_area = area
            largest_face = bbox
    
    return largest_face  # Trả về bounding box của khuôn mặt lớn nhất


def calculate_affine_matrix_from_bbox(bbox, margin, output_size):
    """
    Tính toán ma trận affine dựa trên bounding box của khuôn mặt.
    
    bbox: (x1, y1, x2, y2) bounding box của khuôn mặt
    margin: Khoảng đệm để mở rộng bounding box
    output_size: Kích thước ảnh đầu ra (width, height)
    """
    # Expand bbox with margin
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    scale = max(w, h) * (1 + margin)
    
    # Define source points cho affine transform (3 điểm trong bbox mở rộng)
    src_pts = np.array([
        [cx - scale/2, cy - scale/2],  # Top-left point
        [cx - scale/2, cy + scale/2],  # Bottom-left point
        [cx + scale/2, cy - scale/2],  # Top-right point
    ], dtype=np.float32)

    # Define destination points (output image size)
    dst_pts = np.array([
        [0, 0],                         # Top-left
        [0, output_size[0] - 1],         # Bottom-left
        [output_size[1] - 1, 0],         # Top-right
    ], dtype=np.float32)

    # Affine matrix
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    return M


def warp_image_and_keypoints(image, keypoints, bbox, output_size=(256, 256), margin=0.2):
    """
    Thực hiện affine transform cho ảnh và keypoints.
    
    Parameters:
    - image (np.array): Ảnh gốc.
    - keypoints (np.array): Keypoints (N, 2), dạng (x, y).
    - bbox (np.array): Bounding box (x1, y1, x2, y2).
    - output_size (tuple): Kích thước ảnh đầu ra sau khi resize (default (256, 256)).
    - margin (float): Tỷ lệ margin cho bounding box (default 0.2).
    
    Returns:
    - transformed_image (np.array): Ảnh đã được affine transform.
    - transformed_keypoints (np.array): Keypoints đã được affine transform.
    """
    
    # Expand bbox with margin
    M = calculate_affine_matrix_from_bbox(bbox, margin, output_size)
    
    # Warp ảnh
    transformed_image = cv2.warpAffine(image, M, (output_size[1], output_size[0]),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Warp keypoints (add homogenous coordinate)
    keypoints_homo = np.concatenate([keypoints, np.ones((len(keypoints), 1))], axis=1)
    transformed_keypoints = (M @ keypoints_homo.T).T
    
    return transformed_image, transformed_keypoints


def inverse_transform_keypoints(transformed_keypoints, M):
    """
    Áp dụng ngược affine transform cho keypoints sau khi model dự đoán.
    
    Parameters:
    - transformed_keypoints (np.array): Keypoints dự đoán sau khi affine transform (N, 2).
    - M (np.array): Affine matrix (2x3).
    
    Returns:
    - original_keypoints (np.array): Keypoints đã được transform ngược về ảnh gốc.
    """
    
    # Inverse affine matrix
    M_inv = cv2.invertAffineTransform(M)
    
    # Chuyển đổi keypoints về ảnh gốc
    keypoints_homo = np.concatenate([transformed_keypoints, np.ones((len(transformed_keypoints), 1))], axis=1)
    original_keypoints = (M_inv @ keypoints_homo.T).T
    
    return original_keypoints


