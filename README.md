# 🧠 GAN Explorer — Interactive Generative Adversarial Networks Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-GAN-764BA2?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**An interactive educational dashboard for exploring Generative Adversarial Networks**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hafsa%20Ibrahim-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/)
[![GitHub](https://img.shields.io/badge/GitHub-HafsaIbrahim5-181717?style=flat&logo=github)](https://github.com/HafsaIbrahim5)

</div>

---

## 📋 Overview

**GAN Explorer** is a professional, interactive Streamlit dashboard that makes Generative Adversarial Networks (GANs) accessible, visual, and fun to explore. Whether you're a beginner learning the fundamentals or a practitioner looking for a visual debugging tool, this app has everything you need.

Built from scratch with a sleek **dark purple gradient UI**, animated training, and 8 comprehensive tabs covering every aspect of GANs.

---

## 🚀 Live Demo  
[🔗 View Live App](https://jcyvvdl3n2ze4hfsmagjiy.streamlit.app/)
## ✨ Features

| Tab | Feature |
|-----|---------|
| 📖 **About GANs** | Full theory: concept, math objective, applications, history, training steps, challenges |
| 🎨 **Image Generator** | Generate images from random latent vectors, Latent Grid exploration, download button |
| 🌀 **Interpolation** | Linear + Slerp (Spherical) interpolation between two latent points |
| 📈 **Training Simulator** | Animated or instant loss curves, accuracy plots, G/D ratio, cumulative loss |
| 🏗️ **Architecture** | Dynamic G & D architecture display with parameter counts + PyTorch code |
| 📊 **Analytics Dashboard** | Hyperparameter sensitivity heatmap, loss distributions, batch size comparison |
| 🔬 **Latent Space Explorer** | Distribution violin plots, 2D projection scatter, Slerp interpolation |
| 🆚 **GAN Variants** | Compare 6 GAN variants with pros/cons/use-cases + Radar chart |
| 💡 **Tips & Tricks** | 10 expert training tips + quick reference hyperparameter table |

### 🎛️ Sidebar Controls
- Latent dimension (16–256)
- Training epochs (10–300)
- Batch size (16–256)
- Generator & Discriminator learning rates
- Number of images to generate (4–64)
- Image resolution (16×16 → 64×64)
- Color map selector (8 options)
- Random seed for reproducibility

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/HafsaIbrahim5/gan-explorer.git
cd gan-explorer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📦 Dependencies

```
streamlit>=1.32.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.18.0
```

No GPU required — fully CPU-based simulation!

---

## 🧠 What is a GAN?

A **Generative Adversarial Network** (Goodfellow et al., 2014) consists of two neural networks:

- **Generator G**: Takes random noise `z ~ N(0,I)` and maps it to realistic data
- **Discriminator D**: Classifies real data from generated fakes

They compete in a minimax game:

```
min_G max_D V(D,G) = 𝔼[log D(x)] + 𝔼[log(1 − D(G(z)))]
```

Through training, G learns to produce data indistinguishable from real samples.

---

## 📁 Project Structure

```
gan-explorer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🤝 Connect

- **LinkedIn**: [Hafsa Ibrahim](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/)
- **GitHub**: [HafsaIbrahim5](https://github.com/HafsaIbrahim5)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">
  Built with ❤️ by <strong>Hafsa Ibrahim</strong> — AI / ML Engineer
</div>
