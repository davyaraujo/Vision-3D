import numpy as np
import cv2
import matplotlib.pyplot as plt

print("Version d'OpenCV: ", cv2.__version__)

# --- CONFIGURAÇÃO ---
PATH_IMG = '/home/davy/Ensta/Vision_3D/Images_Homographie/Images_Homographie/'
img1 = cv2.imread(PATH_IMG + "paris_b.jpg")
img2 = cv2.imread(PATH_IMG + "paris_c.jpg") 

points_img1 = []
points_img2 = []

def compute_plane(points):
    point = np.zeros((3,1))
    normal = np.zeros((3,1))
    
    # TODO
    n = np.cross(points[1] - points[0], points[2] - points[0])
    
    n = n / np.linalg.norm(n)
    normal = n.reshape(3,1)
    point = points[0].reshape(3,1)

    return point, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):
    
    indices = np.zeros(len(points), dtype=bool)
    
    # TODO: return a boolean mask of points in range
    dist = np.abs((points - ref_pt.T).dot(normal))
    for i in range(len(points)):
        if dist[i] < threshold_in:
            indices[i] = True
    return indices

def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    # TODO:
    points_indices = np.arange(len(points))
    max_inliers = 0
    N = len(points) 
    for _ in range(NB_RANDOM_DRAWS):
        sample_indices = np.random.choice(N, 3, replace=False)
        sample_points = points[sample_indices]
        ref_pt, normal = compute_plane(sample_points)
        inliers_mask = in_plane(points, ref_pt, normal, threshold_in)
        num_inliers = np.sum(inliers_mask)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_ref_pt = ref_pt
            best_normal = normal
    k = np.log(1 - 0.99) / np.log(1 - (max_inliers / N) ** 3)
    print(f"New best plane with {max_inliers} inliers.")
    print(f"Estimated number of iterations for 99% confidence: {int(np.ceil(k))}")
    return best_ref_pt, best_normal

def select_points(event, x, y, flags, param):
    current_list = param['list'] 
    img_display = param['img']
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(param['window_name'], img_display)
        current_list.append([x, y])
        print(f"Points: {x}, {y}")

print("Step 1 - give the points in the first image")
clone1 = img1.copy()
cv2.namedWindow("Image 1 (Base)")
params1 = {'list': points_img1, 'img': clone1, 'window_name': "Image 1 (Base)"}
cv2.setMouseCallback("Image 1 (Base)", select_points, params1)

while True:
    cv2.imshow("Image 1 (Base)", clone1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Image 1 (Base)")

print("Step 2 - give the exactly same points in the image 2 and after touche q")
clone2 = img2.copy()
cv2.namedWindow("Image 2 (Movel)")
params2 = {'list': points_img2, 'img': clone2, 'window_name': "Imagem 2 (Movel)"}
cv2.setMouseCallback("Image 2 (Movel)", select_points, params2)

while True:
    cv2.imshow("Image 2 (Movel)", clone2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Image 2 (Movel)")

points1_np = np.array(points_img1, dtype=np.float32)
points2_np = np.array(points_img2, dtype=np.float32)

print(f"Points Img1: {len(points1_np)}")
print(f"Points Img2: {len(points2_np)}")

if len(points1_np) != len(points2_np) or len(points1_np) < 4:
    print("Size incorrect")
    exit()

H, status = cv2.findHomography(points2_np, points1_np, cv2.RANSAC, 5.0)
print("Homographie: \n", H)

def stich_images_robust(img_base, img_movel, H):
    h1, w1 = img_base.shape[:2]
    h2, w2 = img_movel.shape[:2]
    
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)
    transformed_corners = transformed_corners.reshape(-1, 2)    
    all_corners = np.concatenate((transformed_corners, [[0, 0], [0, h1], [w1, h1], [w1, 0]]), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    output_img = cv2.warpPerspective(img_movel, H_translation.dot(H), (xmax - xmin, ymax - ymin))

    output_img[translation_dist[1]:h1 + translation_dist[1], 
               translation_dist[0]:w1 + translation_dist[0]] = img_base
    
    return output_img

result = stich_images_robust(img1, img2, H)

img_rectified_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(img_rectified_rgb)
plt.title("Panorama")
plt.axis('off')
plt.show()