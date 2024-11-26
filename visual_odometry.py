# Originally implemented by Nicolai Nielsen
import time, os, cv2
import numpy as np

def transform(R, t):
    T = np.eye(4, dtype = np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T
   
class CameraPoses():
    def __init__(self, data_dir, skip_frames, intrinsic):
        self.K = intrinsic
        
        self.P = self.K @ np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)))
        self.orb = cv2.ORB_create(3000)

        index_params = dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)
        search_params = dict(checks = 50)
        
        self.flann = cv2.FlannBasedMatcher(indexParams = index_params, searchParams = search_params)
        self.current_pose = None
    
    def get_matches(self, img1, img2):
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        # Find matches
        if len(kp1) > 3 and len(kp2) > 3:
            matches = self.flann.knnMatch(des1, des2, k=2)

            # Find the matches that have a low distance
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass
            
            # Draw matches
            img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype = np.uint8)

            q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
            return q1, q2
        else:
            return None, None

    def get_pose(self, q1, q2):
        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.K)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = transform(R, np.squeeze(t))
        
        return transformation_matrix
        
    def decomp_essential_mat(self, E, q1, q2):
        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            # Get the transformation matrix
            T = transform(R, t)
            
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))

            z_sum, scale = sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
            
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return R1, t