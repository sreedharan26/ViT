# Vision Transformer (ViT) – Project README

## 📌 Project Overview
This project is an implementation of the Vision Transformer (ViT) model, based on the paper: ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) presented at ICLR 2021.

## 📷 Objective
Build a Transformer-based model for image classification, replacing traditional CNNs with self-attention mechanisms.

## 🔧 Features
- Image-to-patch conversion (16×16 patches)
- Positional encoding for spatial awareness
- Multi-head self-attention transformer encoder
- Classification using a learnable [CLS] token
- Trained and fine-tuned on the Food-101 dataset

## 🛠️ Tech Stack
- Python
- PyTorch
- NumPy, Matplotlib

## 🧠 Learnings
- Self-attention & Transformer architecture in vision
- Patch embedding & position encoding
- Transfer learning & large-scale pre-training strategies

## 🚀 Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/your-username/vit-project.git
cd vit-project
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision 
```

3. Download the Food-101 dataset and place it in the `data/` folder.
```bash
python3 fetch_data.py
```    

4. Run training:
```bash
python3 main.py
```

## 📜 References
- [An Image is Worth 16x16 Words (ViT Paper)](https://arxiv.org/abs/2010.11929)
- [Google Research GitHub](https://github.com/google-research/vision_transformer)

---

Feel free to contribute or open issues if you find bugs or want to enhance the project!

**Author**: Sree Dharan

