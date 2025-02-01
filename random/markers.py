# Student Name: [Your name here]

# Required libraries:
# - OpenCV (cv2)
# - numpy
# - pickle
import cv2
import numpy as np
import pickle

# Load the camera calibration data
with open("calibration.pckl", "rb") as f:
    cameraMatrix, distCoeffs, _, _ = pickle.load(f)

def main():
    # Read the images
    img1 = cv2.imread("frame-002.png")
    img2 = cv2.imread("frame-253.png")
    
    # Create ArUco dictionary - Update to use new API
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Subtask 1: Detect markers in the first image - Update detection call
    corners, ids, rejected = detector.detectMarkers(img1)
    
    # Draw detected markers
    img1_markers = img1.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img1_markers, corners, ids)
        
        # Subtask 2: Draw coordinate system for each marker
        # Define the marker size parameters
        square_length = 0.026
        marker_length = 0.015
        
        # Create the board object - Update board creation
        board = cv2.aruco.GridBoard((1, 1), marker_length, 
                                  square_length-marker_length,
                                  aruco_dict)
        
        # Draw axes for each marker - Update pose estimation
        for i in range(len(corners)):
            # Create object points for a single marker
            objPoints = np.array([[-marker_length/2, marker_length/2, 0],
                                [marker_length/2, marker_length/2, 0],
                                [marker_length/2, -marker_length/2, 0],
                                [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)
            
            # Get pose for single marker
            retval, rvec, tvec = cv2.solvePnP(objPoints, 
                                             corners[i], 
                                             cameraMatrix, 
                                             distCoeffs)
            
            cv2.drawFrameAxes(img1_markers, cameraMatrix, distCoeffs, rvec, tvec, marker_length)
            
            # Subtask 3: Store pose of marker 1
            if ids[i] == 1:
                marker_1_rvec = rvec
                marker_1_tvec = tvec
    
    # Subtask 4: Draw missing marker in second image
    img2_with_marker = img2.copy()

    # Given pose for marker 1 in second image
    trn = np.array([[0.01428826],
                    [0.02174878],
                    [0.37597986]])
    rot = np.array([[1.576368],
                    [-1.03584672],
                    [0.89579336]])

    # Convert rotation vector to rotation matrix and back to ensure proper format
    rot_matrix, _ = cv2.Rodrigues(rot)
    rot_vec, _ = cv2.Rodrigues(rot_matrix)

    # Create marker image
    marker_size = 100  # pixels
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_img = aruco_dict.generateImageMarker(1, marker_size, marker_img)

    # Define marker corners in 3D space
    marker_points = np.array([
        [-marker_length/2, marker_length/2, 0],
        [marker_length/2, marker_length/2, 0],
        [marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # Project points using OpenCV's projectPoints
    projected_points, _ = cv2.projectPoints(marker_points, rot_vec, trn, cameraMatrix, distCoeffs)
    projected_points = projected_points.reshape(-1, 2)

    # Source points are the corners of the marker image
    source_points = np.array([
        [0, 0],
        [marker_size-1, 0],
        [marker_size-1, marker_size-1],
        [0, marker_size-1]
    ], dtype=np.float32)

    # Get perspective transform
    M = cv2.getPerspectiveTransform(source_points, projected_points)

    # Warp and blend the marker into the image
    warped = cv2.warpPerspective(marker_img, M, (img2.shape[1], img2.shape[0]))
    
    # Create mask and convert to 3 channels
    mask = (warped > 0).astype(np.uint8)
    mask = np.stack([mask] * 3, axis=2)
    mask_inv = 1 - mask

    # Convert warped marker to BGR
    warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # Blend the marker with the original image
    img2_with_marker = img2_with_marker * mask_inv + warped_bgr * mask
    
    # Save or display results
    cv2.imwrite("result_task1.png", img1_markers)
    cv2.imwrite("result_task4.png", img2_with_marker)
    
    # Optional: Display images
    cv2.imshow("Image 1 with markers", img1_markers)
    cv2.imshow("Image 2 with projected marker", img2_with_marker)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 