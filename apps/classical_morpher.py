import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor, Resize, Compose, Normalize

def predict_landmarks_for_classical(image, landmark_predictor_model, device):
    """Predicts landmarks and returns them as an unnormalized tensor for OpenCV."""
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_transformed = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        landmarks = landmark_predictor_model(image_transformed).squeeze(0).cpu()
        # Use 38 landmarks (19 pairs) to match the original LM-1 behavior
        landmarks = landmarks[:38]
        landmarks = landmarks.view(-1, 2)
    return landmarks

def _extract_index_nparray(nparray):
    """Helper function to extract index from numpy where clause."""
    return nparray[0][0] if len(nparray[0]) > 0 else None

def _tensor_to_int_array(tensor):
    """Converts a landmark tensor to a list of integer tuples."""
    return [(int(x[0]), int(x[1])) for x in tensor.numpy()]

def ocular_morph_classical(img1_pil, img2_pil, landmarks1_tensor, landmarks2_tensor):
    """Performs landmark-based morphing using Delaunay triangulation and seamless cloning."""
    img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2BGR)

    points1 = _tensor_to_int_array(landmarks1_tensor)
    points2 = _tensor_to_int_array(landmarks2_tensor)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # --- FIX: Define img2_gray, which was previously missing. ---
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(img1_gray)
    
    points_np = np.array(points1, np.int32)
    convexhull = cv2.convexHull(points_np)
    cv2.fillConvexPoly(mask, convexhull, 255)

    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points1)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        index_pt1 = _extract_index_nparray(np.where((points_np == pt1).all(axis=1)))
        index_pt2 = _extract_index_nparray(np.where((points_np == pt2).all(axis=1)))
        index_pt3 = _extract_index_nparray(np.where((points_np == pt3).all(axis=1)))

        if all(idx is not None for idx in [index_pt1, index_pt2, index_pt3]):
            indexes_triangles.append([index_pt1, index_pt2, index_pt3])
    
    img2_new_face = np.zeros_like(img2)
    
    for triangle_index in indexes_triangles:
        tr1_pt1, tr1_pt2, tr1_pt3 = points1[triangle_index[0]], points1[triangle_index[1]], points1[triangle_index[2]]
        tr2_pt1, tr2_pt2, tr2_pt3 = points2[triangle_index[0]], points2[triangle_index[1]], points2[triangle_index[2]]
        
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x1, y1, w1, h1) = rect1
        cropped_triangle = img1[y1: y1 + h1, x1: x1 + w1]
        points_rel1 = np.array([[tr1_pt1[0] - x1, tr1_pt1[1] - y1], [tr1_pt2[0] - x1, tr1_pt2[1] - y1], [tr1_pt3[0] - x1, tr1_pt3[1] - y1]], np.float32)

        rect2 = cv2.boundingRect(triangle2)
        (x2, y2, w2, h2) = rect2
        points_rel2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2], [tr2_pt2[0] - x2, tr2_pt2[1] - y2], [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.float32)
        
        M = cv2.getAffineTransform(points_rel1, points_rel2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))

        cropped_tr2_mask = np.zeros((h2, w2), np.uint8)
        cv2.fillConvexPoly(cropped_tr2_mask, np.int32(points_rel2), 255)
        
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
        
        img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2: x2 + w2]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        
        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area

    img2_face_mask = np.zeros_like(img2_gray)
    convexhull2 = cv2.convexHull(np.array(points2, np.int32))
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int(x + w / 2), int(y + h / 2))
    
    seamlessclone = cv2.seamlessClone(img2_new_face, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    return cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2RGB)