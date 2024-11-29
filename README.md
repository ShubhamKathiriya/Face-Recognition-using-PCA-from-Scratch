# Face Recognition using Principal Component Analysis (PCA)

This project implements a facial recognition system using **Principal Component Analysis (PCA)**, projecting face images onto a feature space called **Eigenfaces**, which represent variations among distinct faces. The system recognizes faces by comparing them to a pre-existing database and identifying the closest match.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Key Tasks](#key-tasks)
5. [Implementation Details](#implementation-details)
6. [Results and Visualization](#results-and-visualization)
7. [References](#references)

---

## Overview
**Principal Component Analysis (PCA)** is used to reduce the dimensionality of facial image data by identifying the directions (principal components) that account for the most variance. By projecting faces onto a lower-dimensional space (spanned by Eigenfaces), we achieve:
- Efficient representation of facial data.
- Improved computational performance for recognition tasks.

The **Eigenfaces** technique involves:
1. Calculating the mean face vector.
2. Finding eigenvectors of the covariance matrix of the face dataset.
3. Projecting face images onto the space defined by the top eigenvectors.

## Dataset
The project uses the **AT&T Face Dataset**:
- **Description**: Grayscale images of 40 individuals with 10 variations per individual.
- **Image Size**: 92x112 pixels.
- **Variations**: Lighting conditions, facial expressions, and details (e.g., glasses).
- **Dataset Structure**: Images are stored in directories labeled `sX` (X: subject ID), and filenames are `Y.pgm` (Y: image index).
- **Link**: [AT&T Face Dataset](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/)

## Requirements
Install the required libraries using the following command:
```bash
pip install numpy matplotlib opencv-python scikit-learn
```



## Key Tasks
1. **Data Preparation**:
   - Load the AT&T dataset and preprocess images.
   - Split the data into training and test sets.

2. **PCA Implementation**:
   - Compute the mean face.
   - Calculate the covariance matrix.
   - Derive Eigenfaces by solving the eigenvalue problem.
   - Select the top `k` eigenvectors to form the reduced feature space.

3. **Image Reconstruction**:
   - Reconstruct face images by projecting them onto the reduced feature space and back.
   - Visualize differences in reconstruction quality for varying `k`.

4. **Visualization**:
   - Display the mean face and top Eigenfaces.
   - Compare original and reconstructed images.

5. **Face Recognition**:
   - Perform face recognition by comparing test images to training data.
   - Evaluate recognition accuracy for different values of `k`.

## Implementation Details

### Data Preprocessing
- Images are reshaped into vectors of size `10304` (92x112).
- Training and test sets are created by splitting the dataset (e.g., 80% training, 20% testing).

### PCA Algorithm
1. **Compute the Mean Face**:
   - Calculate the average of all training face vectors.
2. **Center the Data**:
   - Subtract the mean face from each image vector.
3. **Covariance Matrix**:
   - Compute the covariance matrix of the centered data.
4. **Eigenfaces**:
   - Calculate eigenvectors and eigenvalues of the covariance matrix.
   - Select the top `k` eigenvectors corresponding to the largest eigenvalues.

### Face Recognition
- Each face is represented by its projection coefficients in the Eigenface space.
- Recognition is performed by finding the Euclidean distance between test and training projections.

### Evaluation Metrics
- **Accuracy**: Measure of correctly identified faces across different `k` values.
- **Reconstruction Error**: Comparison of original and reconstructed images.

## Results and Visualization
1. **Mean Face**:
   - Visual representation of the average face across the training set.
2. **Eigenfaces**:
   - Top principal components representing the largest variations.
3. **Reconstructed Images**:
   - Side-by-side comparison of original and reconstructed images.
4. **Recognition Accuracy**:
   - Accuracy values for varying `k` principal components.


## References
1. Turk, M. A., & Pentland, A. P. (1991). *Eigenfaces for Recognition*. Proceedings of CVPR. [Paper Link](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)
2. AT&T Laboratories Cambridge. *The ORL Database of Faces*. [Dataset Link](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/)
