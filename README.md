# Segmenting identified fracture families from 3D fracture networks in Montney rock using a deep learning-based method
![400dpi](https://github.com/user-attachments/assets/f747526b-6f60-409f-96b7-a350407ca767)

Link to video and point cloud dataset of the digital fracture network: https://geogroup.utoronto.ca/li-supplementary-material/

If you use the code here in your research, please cite this paper:

Li M, Grasselli G. 2025. Segmenting identified fracture families from 3D fracture networks in Montney rock using a deep learning-based method. Journal of Rock Mechanics and Geotechnical Engineering.

# Overview
This release include the demonstration code for training and evaluation, ZY dataset, and the learned ZY model.
# How to use
How to train a new model

    python main.py --isTrain

The trained model and predicted image results are saved under folder Checkpoint/. Our trained model to segment fractures in ZY images is provided: Checkpoint/Trained_ZY_segment_model/Trained_ZY_segment_model.pt

How to test

    python main.py --model_to_load "Trained_ZY_segment_model"


