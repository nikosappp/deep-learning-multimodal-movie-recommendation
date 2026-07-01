# Multimodal Deep Learning for Movie Recommendation

This repository contains the lab project developed for the **Neural Networks and Deep Learning** course at the School of **Electrical and Computer Engineering (ECE), National Technical University of Athens (NTUA)**.

## Assignment Overview
This project implements a hybrid movie recommendation system by leveraging a multimodal deep learning framework that integrates Computer Vision (CV), Natural Language Processing (NLP), and Graph Neural Networks (GNNs). The system uses content-based features extracted from both visual and textual modalities to initialize a graph structure, which is then used to perform link prediction for user-movie recommendations.

The implementation is divided into three main tasks:

### Task 1: Computer Vision (CV) - Visual Embeddings
* Objective: Multi-label movie genre classification based on movie poster images.
* Approach: Compares two approaches: a custom Convolutional Neural Network (CNN) trained from scratch and fine-tuning a pre-trained model on movie posters. After comparing their performance, dense visual embeddings are extracted from the penultimate layer of the best-performing model to use in the final recommendation graph.
* Transform Pipeline: Input images are scaled, center-cropped to a square 128x128 resolution, and standardized using ImageNet normalization constants to stabilize training.
* Data Imbalance: A smoothed positive weight adjustment (`pos_weight`) is incorporated into the loss function to penalize misclassifications on rare genres and alleviate severe label skew.

### Task 2: Natural Language Processing (NLP) - Textual Embeddings
* Objective: Multi-label movie genre classification based on textual movie plot summaries.
* Approach: Trains a Recurrent Neural Network (RNN) or a Transformer-based architecture on textual data.
* Feature Extraction: Semantic text embeddings are extracted to capture the narrative and thematic context of the movies.

### Task 3: Graph Neural Networks (GNN) - Link Prediction
* Objective: Recommend movies to users by framing recommendation as a link prediction task.
* Graph Construction: A bipartite user-item graph is constructed using the `ml-latest-small` MovieLens dataset.
* Initialization & Training: Movie nodes are initialized by fusing the visual and textual dense embeddings extracted from Tasks 1 and 2. A Graph Neural Network is then trained on this multimodal graph to predict user-movie interaction links.

---

## Dataset
* **Movie Metadata and Posters:** Used for training the CV and NLP multi-label classification models.
* **MovieLens (ml-latest-small):** Used to build the bipartite user-movie graph and train the GNN for recommendation.

---

## Environment and Requirements
This project was implemented and executed inside a Google Colab environment. The primary libraries used include:
* PyTorch
* PyTorch Geometric (PyG)
* Torchvision
* Scikit-Learn
* Pandas, Numpy, and Matplotlib

---

## Usage
The entire pipeline is contained within the provided Jupyter Notebook (`DL_LabProject.ipynb`). 

To view or run the notebook directly in Google Colab, click the **Open in Colab** badge located at the top of the notebook file within this repository. 
