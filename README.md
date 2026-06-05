# FALCON: Federated Autoencoder-Based IDS

FALCON stands for **Federated Autoencoder-Based Latent Consensus Outlier-Resilient Network**. It is a privacy-preserving and poison-resilient Intrusion Detection System (IDS) for cyber threat detection.

The system uses **Federated Learning**, **Deep Autoencoders**, and **anomaly detection** to detect malicious network traffic without centralizing raw client data.

## Key Features

- Privacy-preserving IDS using Federated Learning
- Autoencoder-based anomaly detection
- CIC-IDS2017 network intrusion dataset
- Feature reduction from 78 to 21 high-relevance features
- MSE reconstruction error for threat detection
- Latent Consensus mechanism to reject poisoned client updates
- Quality Rollback to prevent degraded global model updates
- Flask, Streamlit, and REST API-based live system demo

## Tech Stack

Python, Scikit-learn, Pandas, NumPy, Flask, Streamlit, REST APIs, Federated Learning, Autoencoders, Network Intrusion Detection

## System Overview

FALCON has three main parts:

1. **Local Edge Clients**  
   Train autoencoder models locally on benign network traffic.

2. **Global Server**  
   Aggregates validated client model updates without receiving raw traffic data.

3. **Security Layer**  
   Uses Latent Consensus and Quality Rollback to reject suspicious or performance-degrading updates.

## Dataset and Preprocessing

The project uses the **CIC-IDS2017** dataset. The preprocessing pipeline includes:

- Data cleaning
- Feature selection
- Log transformation
- MinMax scaling
- Reduction from 78 features to 21 features
- Benign/attack traffic separation

## Results

The federated model achieved a stable global F1-score of:

```text
0.8450
