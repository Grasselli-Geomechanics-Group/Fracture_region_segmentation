# Fracture_region_segmentation
This repository is provided as the supplementary document of our research paper to reproduce the fracture regions segmentation results.

# How to use
How to train a new model

    python main.py --isTrain

The trained model and predicted image results are saved under folder Checkpoint/. Our trained model to segment fractures in ZY images is provided: Checkpoint/Trained_ZY_segment_model/Trained_ZY_segment_model.pt

How to test

    python main.py --model_to_load "Trained_ZY_segment_model"


