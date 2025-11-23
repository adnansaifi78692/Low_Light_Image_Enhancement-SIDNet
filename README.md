# 🔦 Low-Light Image Enhancement using SIDNet

A deep learning–based solution to enhance extremely low-light images using a **SIDNet architecture**, trained on the **LOLv2 dataset**. This project restores illumination, contrast, and perceptual quality using a supervised learning pipeline powered by multiple loss functions and quantitative metrics.

---

## 🚀 Project Overview
Low-light images suffer from **noise, low contrast, color distortion, and reduced visibility**, affecting applications such as surveillance, mobile photography, medical imaging, and autonomous driving.

Traditional enhancement methods fail under extreme lighting.  
This project uses **SIDNet — a lightweight encoder–decoder CNN with residual blocks** — to learn complex mappings between low-light images and their corresponding well-lit targets.

---

## 🎯 Objectives
- Enhance visibility and perceptual quality of low-light images
- Preserve structure, texture, and color balance
- Train using paired low-light and ground-truth images (LOLv2)
- Evaluate performance using full- and no-reference image quality metrics

---

## 🧠 SIDNet Architecture
The network follows an encoder–decoder structure with skip connections:

Input
 ↓
Conv (3→32)
 ↓
Downsample #1 (32→64)
 ↓
Residual Block (64)
 ↓
Downsample #2 (64→128)
 ↓
2x Residual Blocks (128)
 ↓
Upsample #1 (128→64) + skip connection from earlier 64
 ↓
Residual Block (64)
 ↓
Upsample #2 (64→32) + skip connection from earlier 32
 ↓
Residual Block (32)
 ↓
Conv (32→3)
 ↓
ReLU (limit output to [0, 1])


🔹 Skip connections preserve high-frequency spatial features  
🔹 Residual learning stabilizes gradients and improves detail recovery

---

## 🗃 Dataset — **LOLv2**
The model is trained on the **Real-Captured subset** of the LOLv2 benchmark dataset.

| Type | Description |
|------|-------------|
| Real-Captured | Real low-light & well-lit image pairs |
| Synthetic | Degraded clean images (not used for training here) |

Dataset includes scenes with varying noise levels, indoor lighting, chromatic imbalance, and dark exposure.

---

## ⚙ Training Methodology
| Component | Method |
|----------|--------|
| Framework | PyTorch |
| Training Strategy | Patch-based learning (256×256) |
| Device | GPU-accelerated |
| Optimization | Adam (LR = 3e-4) |
| Epochs | 100 |

### 🔥 Total Loss Function
1.0 × MSE +
1.0 × SSIM +
0.3 × Exposure +
0.5 × Color Constancy +
0.1 × VGG Perceptual

yaml
Copy code

This balances **pixel accuracy + structure + perceptual realism + natural lighting**.

---

## 📏 Evaluation Metrics
| Metric | Type | Goal |
|--------|------|-----|
| PSNR | Full-reference | Higher is better |
| SSIM | Full-reference | Higher is better |
| LPIPS | Full-reference | Lower is better |
| NIQE | No-reference | Lower is better |

---

## 📊 Results
| Image | PSNR ↑ | SSIM ↑ | NIQE ↓ | LPIPS ↓ |
|-------|--------|--------|--------|---------|
| 1 | 20.37 | 0.8792 | 3.9754 | 0.1833 |
| 2 | 21.80 | 0.8965 | 3.8688 | 0.1635 |
| 3 | 21.65 | 0.8147 | 5.0269 | 0.2479 |
| 4 | 22.25 | 0.8360 | 3.2803 | 0.2570 |

---

## 🧪 Usage

### 🔹 Setup

pip install -r requirements.txt

🔹 Train
bash
Copy code
python low_light.py
🔹 Test & Evaluate
bash
Copy code
python test.py
📂 Folder Structure
cpp
Copy code
📦 Low_Light_Image_Enhancement-SIDNet
│── README.md
│── requirements.txt
│── LICENSE
│
├── model/
│   └── SIDNet.py
├── scripts/
│   ├── train.py  (low_light.py)
│   └── test.py
├── docs/
│   ├── Dissertation.pdf
│   └── Presentation.pptx
├── samples/
│   ├── sample_1.png
│   ├── sample_2.png
│   └── sample_3.png
└── ckpt/ (optional)
⚠ Challenges & Limitations
High computational demand; training requires GPU

Overexposure can occur in extremely bright regions

Generalization to outdoor night scenes needs improvement

Real-time performance not yet optimized for mobile devices

🔮 Future Scope
Lightweight mobile-deployable version

Joint enhancement + noise removal + color restoration

Unsupervised/self-supervised training (no paired dataset required)

Domain generalization for outdoor night surveillance

👤 Author
Mohd Adnan Saifi
M.Tech (Computer Science)

🧾 Citation
powershell
Copy code
Saifi, M.A. (2025). Low-Light Image Enhancement using SIDNet. GitHub Repository.
