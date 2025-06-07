# Bridging Black-Box and No-Box: Embedding Reconstruction Attacks on Deep Recognition Systems
This repository contains the official implementation of the paper:  
**Bridging Black-Box and No-Box: Embedding Reconstruction Attacks on Deep Recognition Systems**.
The codes will be published after being accepted.
## Overview

In this paper, we study adversarial scenarios requiring less prior knowledge than black-box attacks and propose a four-tier ERA framework, which progressively decreases adversarial knowledge while increasing attack complexity from black-box to no-box settings.

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
