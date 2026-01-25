
import json
import sys
import cv2
import numpy as np

def solve_captcha(captcha_path, logo_path, output_path, threshold=0.8):
    # Load the captcha and logo images
    captcha_img = cv2.imread(captcha_path)
    captcha_gray = cv2.cvtColor(captcha_img, cv2.COLOR_BGR2GRAY)
    logo_img = cv2.imread(logo_path, 0)
    w, h = logo_img.shape[::-1]

    # Perform template matching
    res = cv2.matchTemplate(captcha_gray, logo_img, cv2.TM_CCOEFF_NORMED)
    
    # Find matches above the threshold
    loc = np.where(res >= threshold)
    
    # Cluster the points
    points = list(zip(*loc[::-1]))
    clusters = []
    for pt in points:
        found_cluster = False
        for cluster in clusters:
            # Check if the point is close to any point in the cluster
            if any(np.linalg.norm(np.array(pt) - np.array(c_pt)) < 20 for c_pt in cluster):
                cluster.append(pt)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([pt])
            
    # Get the center of each cluster
    centers = []
    for cluster in clusters:
        center = np.mean(cluster, axis=0, dtype=int)
        centers.append(tuple(center))

    # Draw rectangles around the cluster centers
    for pt in centers:
        cv2.rectangle(captcha_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        
    # Save the result
    cv2.imwrite(output_path, captcha_img)
    
    return centers

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_template_matching.py <captcha_image_path> <logo_image_path>")
        sys.exit(1)
    captcha_image = sys.argv[1]
    logo_image = sys.argv[2]
    output_image = "captcha_logs/captcha_solved.png"
    
    centers = solve_captcha(captcha_image, logo_image, output_image)
    print(json.dumps(centers))
