CPU vs GPU Acceleration of CNN-Based Face Recognition
A Computer Architecture Performance Analysis

# 1. Project Overview

This project aims to develop a deep-learning-based face recognition system that identifies registered individuals from new face images.

The primary objective is to analyse the computational performance of CNN-based face recognition when executed on a CPU versus a GPU.

The project combines Computer Vision, Deep Learning, Parallel Processing and Computer Organization and Architecture (COA) concepts to understand how hardware architecture affects neural network performance.

# 2. Problem Statement

Face recognition involves numerous mathematical computations performed by deep neural networks. As the workload increases, computational requirements also increase.

CPUs and GPUs have different processing architectures. This project investigates how these architectural differences influence the execution of CNN-based face recognition workloads.

# 3. Project Objectives

- Develop a functional face recognition application.
- Implement CNN-based facial feature extraction.
- Generate and compare face embeddings.
- Execute the same CNN workload on CPU and GPU, where supported.
- Measure inference time, throughput and memory usage.
- Visualise COA concepts such as memory hierarchy, cache, registers and parallel processing.
- Analyse CPU and GPU performance through experimental results.

# 4. Key Features

- Student/person face registration.
- Face image upload and processing.
- CNN-based face recognition.
- Face embedding generation.
- Similarity-based matching.
- Unknown face detection.
- CPU benchmarking.
- GPU benchmarking (subject to hardware availability).
- CPU vs GPU performance comparison.
- COA execution visualization.
- Performance graphs and analytics.

# 5. System Workflow

Camera / Image Input
↓
Face Detection
↓
Image Preprocessing
↓
CNN Feature Extraction
↓
Face Embedding Generation
↓
Similarity Matching
↓
Identity / Unknown

The same CNN workload will subsequently be evaluated using CPU and GPU execution.

# 6. Technology Stack

- **Frontend:** React, HTML, CSS, JavaScript
- **Backend:** Python, FastAPI
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV
- **Database:** SQLite
- **Visualisation:** Chart.js
- **Version Control:** Git and GitHub

*Technologies may be refined during development.*

# 7. COA Concepts Covered

- CPU architecture
- GPU parallelism
- Memory hierarchy
- Cache hit and cache miss
- Registers
- Computation units
- Data movement
- Latency and throughput
- Performance benchmarking

The COA visualization will provide a conceptual representation of processor operations rather than claiming direct access to physical cache and register activity.

8. Development Roadmap

- [ ] Frontend development
- [ ] Backend development
- [ ] Database integration
- [ ] Face registration
- [ ] Face recognition implementation
- [ ] CPU benchmarking
- [ ] GPU benchmarking
- [ ] COA visualization
- [ ] Performance analytics
- [ ] Testing and deployment

9. Expected Outcome

The final application will provide a working face recognition demonstration along with an experimental comparison of CPU and GPU execution.

The objective is to understand how computer architecture influences the computational performance of deep-learning workloads.

## 10. Project Status

🚧 Currently in development.
