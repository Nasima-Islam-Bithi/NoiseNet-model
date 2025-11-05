# NoiseNet: CNN-Based Regression for Gaussian Noise Level Estimation

Predict the standard deviation (σ) of additive white Gaussian noise (AWGN) in color images using a VGG16–based convolutional regressor with skip connections and a fully connected head.

---

## 🔧 Key Features
- **Backbone:** Pretrained VGG16 feature extractor (`features[0:10]`).
- **Head:** Extra conv blocks + skip connection + MLP regressor for σ ∈ [0.01, 0.20].
- **Synthetic labels:** σ sampled uniformly in `[0.01, 0.20]`; AWGN applied on-the-fly.
- **Metrics:** MSE (loss), MAE, RMSE, R², MAPE, optional RMSLE.
- **Train/val split:** Random 70/30 over images.
- **Hardware-aware:** Runs on GPU if available (`device='cuda'`).

---

## 🗂️ Repository Structure (suggested)
```
.
├── train.py                 # (optional) main script if you split files
├── model.py                 # (optional) model class if you split files
├── README.md                # this file
└── data/
    └── images/             # your RGB images (jpg/png/jpeg)
```
> In the provided script, you set `image_dir` directly to your dataset folder.

---

## 🧠 Model Architecture (high-level)
- **VGG16 features (frozen)** → **Conv(128→256) → Conv(256→128)** → **Conv(128→64)**  
- **Skip path:** `1×1 Conv(128→64)` from VGG features  
- **Fusion:** element-wise `add` of skip and main path  
- **Regressor (MLP):** `Flatten(64×56×56) → 4096 → 2048 → 1024 → 1` with LeakyReLU, LayerNorm, Dropout.

> ⚠️ **Parameter count:** ~833M parameters due to the large fully-connected layers after flattening (64×56×56 = 200,704 → 4096). Consider adding **adaptive pooling** (e.g., `AdaptiveAvgPool2d((7,7))`) to reduce parameters if memory is tight.

---

## 📦 Requirements
- Python 3.9+
- PyTorch, torchvision
- scikit-learn
- Pillow
- tqdm
- matplotlib
- tensorboard (optional)

Install:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose CUDA/CPU as needed
pip install scikit-learn pillow tqdm matplotlib tensorboard
```

> ℹ️ **Torchvision note:** In newer versions, use the weights enum instead of `pretrained=True`, e.g.:
> ```python
> vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
> ```
> The current script suppresses the deprecation warnings with `warnings.filterwarnings`.

---

## 📁 Data
Point `image_dir` to a folder containing RGB images:
```python
image_dir = r'D:/.../coco_train/train2017'
```
Valid extensions: `.png`, `.jpg`, `.jpeg`.

**Labels:** Generated on-the-fly. For each image:
- Sample `σ ~ U(0.01, 0.20)`
- Create `noisy = clean + N(0, σ²)`, clipped to `[0,1]`
- Return `(noisy_tensor, σ)`

---

## ▶️ Quick Start

1) **Set your dataset path** in the script:
```python
image_dir = r'path/to/your/images'
```

2) **Run training** (from your Python environment):
```bash
python your_script.py
```
This will:
- Split images into **70% train / 30% val**
- Train for **25 epochs** with Adam (lr=1e-4, wd=1e-4), StepLR (step=5, γ=0.1)
- Save weights to `test12_model_coco.pth`

3) **Monitor (optional)** with TensorBoard (if you enable the writer lines):
```bash
tensorboard --logdir runs
```

---

## 📊 Reported Example Metrics (from logs)
*(These will vary with dataset & hardware.)*
- After a few epochs, validation typically achieves:
  - **MAE ≈ 0.003–0.005**
  - **RMSE ≈ 0.004–0.006**
  - **R² ≈ 0.99+**
  - **MAPE ≈ 4–6%**

Example (Epoch 10, Val):
```
Test Loss (MSE): 0.0000, MAE: 0.0034, R2: 0.9933, MAPE: 5.1600, RMSE: 0.0045
```

---

## 🏗️ Code Walkthrough

### Model
```python
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)  # or weights=VGG16_Weights.IMAGENET1K_FEATURES
        self.features = nn.Sequential(vgg16.features[0:10])

        self.conv_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.ReLU())  # skip

        self.regressor = nn.Sequential(
            nn.Linear(64*56*56, 4096), nn.LeakyReLU(0.01), nn.LayerNorm(4096), nn.Dropout(0.3),
            nn.Linear(4096, 2048),    nn.LeakyReLU(0.01), nn.LayerNorm(2048), nn.Dropout(0.3),
            nn.Linear(2048, 1024),    nn.LeakyReLU(0.01),
            nn.Linear(1024, 1),
        )
```
- VGG features are **frozen** by setting `requires_grad=False`.

### Dataset
- Loads RGB images, resizes to `224×224`, converts to tensor in `[0,1]`.
- Samples a random `σ` per image, adds AWGN, returns `(noisy, σ)`.

### Training / Evaluation
- **Loss:** `nn.MSELoss()`
- **Optimizer:** Adam (`lr=1e-4`, `weight_decay=1e-4`)
- **Scheduler:** `StepLR(step_size=5, gamma=0.1)`
- **Metrics:** MAE, RMSE, R², MAPE, (RMSLE guarded against negatives)

---

## 🚀 Inference
To estimate σ for a single image:
```python
from PIL import Image
import torch
from torchvision import transforms

model = CustomCNN().to(device)
model.load_state_dict(torch.load('test12_model_coco.pth', map_location=device))
model.eval()

tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
img = tfm(Image.open('path/to/image.jpg').convert('RGB')).unsqueeze(0).to(device)
with torch.no_grad():
    pred_sigma = model(img).item()
print(f"Estimated σ: {pred_sigma:.4f}")
```

---

## 🧪 Reproducibility Tips
- Set seeds:
```python
import torch, numpy as np, random
torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
- Fix train/val split (`random_state=42` is already used).

---

## 🧰 Practical Notes & Troubleshooting
- **Memory usage:** The large fully connected layers can be heavy (model ~3.3 GB est.). If you hit OOM:
  - Add `nn.AdaptiveAvgPool2d((7,7))` before flattening.
  - Replace the MLP with a lighter head (e.g., GAP → 512 → 1).
  - Use smaller batch size.
  - Use **mixed precision** (`torch.cuda.amp`) and **gradient accumulation**.
- **RMSLE NaN:** This is expected if predictions/targets are ≤0. We already guard it in code.
- **Torchvision warnings:** Use weights enum to silence deprecation and remove filters.
- **Image value range:** After noise, images are clipped to `[0,1]`. Confirm `X.max()==1`, `X.min()==0` in loaders.

---

## 📈 Extending the Work
- Train on **real** noisy/clean pairs; move from synthetic to real-world distribution.
- Predict **both** Gaussian and Poisson parameters (e.g., (σ, λ)).
- Test robustness on **mixed noise** (e.g., Gaussian + JPEG artifacts).
- Add **data augmentations** (flips, color jitter) before noise injection.
- Try alternative backbones (e.g., ResNet-18/34) for a lighter model.

---

## 💾 Saving & Loading
- Save after training:
```python
torch.save(model.state_dict(), 'test12_model_coco.pth')
```
- Load for inference:
```python
state = torch.load('test12_model_coco.pth', map_location=device)
model.load_state_dict(state)
model.eval()
```

---

## 📜 License
This project is provided as-is for research and educational use. Add your preferred license (e.g., MIT) here.

---

## 🙌 Acknowledgments
- VGG16 backbone from `torchvision.models`.
- Div2K/Flickr2K/COCO images used as base clean images (per your local setup). Please respect their licenses.
- Training loops inspired by standard PyTorch patterns.

---

## 🧾 Citation (example)
If you use this code in academic work, please cite your paper and this repository:
```
@misc{noisenet2025,
  title={NoiseNet: CNN-Based Regression for Gaussian Noise-Level Estimation in Color Images},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourrepo/noisenet}}
}
```
