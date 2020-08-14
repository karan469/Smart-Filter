# Smart Filter

This is the repository for Smart-Filter, a deep learning open source web app. This repository provides end-to-end pipeline from model architecture of face detection using transfer learning to deployment of a web application using Docker container.  
  
**Visit the official repository webpage:** https://karan469.github.io/Smart-Filter/

**Web app soon to be deployed.**

## Requirements
- tensorflow-cpu==2.2
- pytorch==1.6.0+cpu: pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
- detectron2-cpu (for inference): python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html
- detectron2-gpu (for training): python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
- CUDA (10.1)

*Caution*: Specified softwares must be installed in their correct versions. Mentioned versions are compatible with each other. 

## Subdirectories

**Deployment/**  
Web application for deploying a trained set of models on Flask using Docker containers. To read more about Docker, visit ![official Docker documentation](https://docs.docker.com/).
  - templates/
  - uploads/
  - app.py
  - detectron.py
  - smile.py
  - utils.py
  - facedetector.py
  - requirements.txt

**training/**  
Contains training modules for semantic segmentation, face detection and smile detection. iPython notebooks contains relevant model architectures.
  - Face Detection using Detectron2/
  - Final Ensemble/
  - Key Points Detection/
  - Semantic Segmentation/
  - Smile Detection

**results/**  
Contains demo images as examples of working prototype of final web application consisting of features such as custom background and caption writing according to facial features such as smile.

## Contact  
For more information visit the official documentations of different frameworks.  
For any queries, email ![Karan Tanwar](mailto://tkaran.iitd@gmail.com).
