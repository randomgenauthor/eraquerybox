# Bridging Black-Box and No-Box: Embedding Reconstruction Attacks on Deep Recognition Systems
This repository contains the official implementation of the paper:  
**Bridging Black-Box and No-Box: Embedding Reconstruction Attacks on Deep Recognition Systems**.
The codes will be published after being accepted.
## Overview

HyFIDS is a lightweight Intrusion Detection System (IDS) designed specifically for Internet of Vehicles (IoV) scenarios.  
It integrates spatial and frequency domain features to efficiently detect anomalies in both intra-vehicle and inter-vehicle networks, achieving high detection performance with minimal computational overhead.

Key Features:
- Hybrid feature extraction combining raw byte embeddings and 2D Fourier representations.
- Extremely low FLOPs and model size for resource-constrained IoV devices.
- Competitive accuracy compared to state-of-the-art (SOTA) methods.

## Project Structure

```
erqquerybox/
├── dataset/          # Sample datasets and data processing scripts
├── model/            # Model structure and training
├── deploy/           # Deployment simulation program
├── requirements.txt  # Python dependencies
└── README.md         # Project description
```

## Installation

Clone this repository:
```bash
git clone https://github.com/randomgenauthor/eraquerybox.git
cd eraquerybox
```

Create a virtual environment (optional but recommended):
```bash
python3 -m venv eraquerybox-env
source eraquerybox-env/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Preparation
Before reproducing our work, you need to download the basic codes from https://github.com/foivospar/Arc2Face and https://github.com/HaiyuWu/Vec2Face.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
