# Vision-3D
 This is a solid set of scripts covering fundamental 3D vision and image processing techniques. Since you have three distinct but related files RANSAC for 3D point clouds, Homography for image rectification, and Panorama for image stitching that ties them all together under a "Computer Vision &amp; 3D Modeling" theme.

 # 3D Vision & Geometric Modeling Toolbox

This repository contains a collection of Python scripts developed for 3D modeling and computer vision tasks. The project focuses on geometric transformations, robust estimation algorithms (RANSAC), and image processing.

## 🚀 Features

### 1. 3D Plane Detection (RANSAC)
Located in `RANSAC.py`, this module processes 3D point clouds (PLY format) to detect and extract geometric planes.
* **Core Logic:** Implements the RANSAC (Random Sample Consensus) algorithm from scratch.
* **Multi-Plane Extraction:** Features a `multi_RANSAC` function to iteratively identify and segment multiple surfaces within a single scene.
* **Mathematical Foundation:** Computes plane normals and point-to-plane distances to classify inliers.

### 2. Image Homography & Rectification
Located in `homography.py`, this script allows users to perform planar homography transformations.
* **Interactive Selection:** Users can manually select reference points on an image.
* **Direct Linear Transform (DLT):** Implements the SVD-based approach to solve for the homography matrix $H$.
* **Rectification:** Transforms warped perspectives (e.g., a photo of a painting taken at an angle) into a front-parallel view.

### 3. Panorama Stitching
Located in `panorama.py`, this script combines multiple images into a single wide-angle panorama.
* **Robust Matching:** Uses manual point correspondence combined with `cv2.findHomography` and RANSAC to handle outliers.
* **Perspective Warping:** Features a `stich_images_robust` function that calculates the necessary canvas size and translation to prevent image cropping during warping.
* **Visualization:** Uses Matplotlib for high-quality rendering of the final stitched result.

## 🛠️ Requirements

Ensure you have the following dependencies installed:

```bash
pip install numpy opencv-python matplotlib
