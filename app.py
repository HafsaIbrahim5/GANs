"""
╔══════════════════════════════════════════════════════════════════╗
║          GAN Explorer — Generative Adversarial Networks          ║
║                  Built by Hafsa Ibrahim                          ║
║  LinkedIn: https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/     ║
║  GitHub  : https://github.com/HafsaIbrahim5                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile
import json

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GAN Explorer — Hafsa Ibrahim",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/",
        "Report a bug": "https://github.com/HafsaIbrahim5",
        "About": "## 🧠 GAN Explorer\nBuilt by **Hafsa Ibrahim** — AI/ML Engineer",
    },
)

# ══════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d0b1e 0%, #1a1040 35%, #0d1b2a 70%, #0a0a1a 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #13103a 50%, #0d1b2a 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.25);
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color .3s, box-shadow .3s;
}
.card:hover {
    border-color: rgba(139,92,246,0.45);
    box-shadow: 0 8px 32px rgba(102,126,234,0.12);
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #ec4899 100%);
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 25px 80px rgba(102,126,234,0.35);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='20'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero h1 { font-size: 2.9rem; font-weight: 900; color: white; margin: 0; position: relative; }
.hero p  { font-size: 1.15rem; color: rgba(255,255,255,0.88); margin-top:.5rem; position: relative; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50px;
    padding: .3rem 1rem;
    font-size: .85rem;
    color: white;
    margin-top: .8rem;
    position: relative;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(102,126,234,.2); }
.metric-val { font-size: 2.1rem; font-weight: 800; color: #a78bfa; line-height: 1; }
.metric-lab { font-size: .78rem; color: #94a3b8; margin-top: .35rem; }
.metric-icon { font-size: 1.6rem; margin-bottom: .3rem; }

/* ── Section titles ── */
.section-title {
    font-size: 1.55rem; font-weight: 800;
    background: linear-gradient(90deg, #667eea 0%, #a78bfa 60%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1.2rem; margin-top: .5rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px; padding: .35rem;
    gap: .2rem;
    border: 1px solid rgba(139,92,246,0.15);
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    border-radius: 10px !important;
    padding: .5rem 1.1rem !important;
    font-weight: 500 !important;
    font-size: .9rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important; border: none !important;
    border-radius: 12px !important;
    padding: .65rem 1.8rem !important;
    font-weight: 700 !important; font-size: .92rem !important;
    transition: all .3s !important;
    box-shadow: 0 4px 15px rgba(102,126,234,.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(102,126,234,.5) !important;
}

/* ── Sliders / Selectboxes ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #667eea, #a78bfa) !important;
}

/* ── Code blocks ── */
code { font-family: 'JetBrains Mono', monospace !important; }

/* ── Divider ── */
hr { border-color: rgba(139,92,246,0.2) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(102,126,234,0.08) !important;
    border-radius: 10px !important;
}

/* ── Info/Warning boxes ── */
.stAlert { border-radius: 12px !important; }

/* ── Footer ── */
.footer {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    color: #94a3b8;
    font-size: .88rem;
    margin-top: 3rem;
}
.footer a {
    color: #a78bfa; text-decoration: none;
    margin: 0 .6rem; font-weight: 600;
    transition: color .2s;
}
.footer a:hover { color: #ec4899; }

/* ── Social link buttons ── */
.social-btn {
    display: flex; align-items: center; gap: .5rem;
    padding: .55rem 1rem; border-radius: 10px;
    text-decoration: none; font-size: .85rem;
    font-weight: 600; margin-bottom: .5rem;
    transition: opacity .2s, transform .2s;
    color: white !important;
}
.social-btn:hover { opacity: .9; transform: translateX(3px); }
.linkedin-btn { background: linear-gradient(135deg, #0077b5, #005885); }
.github-btn   { background: linear-gradient(135deg, #2d333b, #161b22); border: 1px solid #444c56; }

/* ── Step number badges ── */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2rem; height: 2rem; border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; font-weight: 900; font-size: .85rem;
    margin-bottom: .5rem;
}

/* ── Tip cards ── */
.tip-card {
    background: rgba(255,255,255,0.03);
    border-radius: 14px; padding: 1.1rem 1.2rem;
    margin-bottom: .8rem;
    transition: transform .2s, box-shadow .2s;
}
.tip-card:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,.2); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,.03); }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,.6); }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════


def generate_noise(batch_size, latent_dim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(batch_size, latent_dim)


def simple_generator(noise, img_size=28):
    """Simulates a trained GAN generator — produces structured patterns from noise."""
    imgs = []
    for z in noise:
        seed = int(abs(z[0] * 1e6)) % (2**31 - 1)
        rng = np.random.RandomState(seed)
        img = rng.rand(img_size, img_size) * 0.2
        cx = int(img_size * np.clip(0.25 + 0.5 * abs(z[1] % 1), 0.1, 0.9))
        cy = int(img_size * np.clip(0.25 + 0.5 * abs(z[2] % 1), 0.1, 0.9))
        Y, X = np.ogrid[:img_size, :img_size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # Add two Gaussian blobs for variety
        blob1 = np.exp(-(dist**2) / (2 * (img_size * 0.15) ** 2))
        blob2 = np.exp(
            -((X - (img_size - cx)) ** 2 + (Y - (img_size - cy)) ** 2)
            / (2 * (img_size * 0.1) ** 2)
        )
        weight = 0.5 + 0.5 * np.sin(z[3] if len(z) > 3 else 0)
        img = img + 0.6 * blob1 + 0.35 * blob2 * weight
        img = np.clip(img, 0, 1)
        imgs.append(img)
    return np.array(imgs)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mlp_generator_from_npz(noise, weights, img_size):
    """
    Inference-only MLP generator using NumPy weights loaded from .npz.
    Expected keys (minimum): W1, b1, W2, b2 (can be more layers: W3, b3, ...).
    Output layer must produce img_size*img_size features per sample.
    """
    x = noise.astype(np.float32)

    layer = 1
    while True:
        w_key = f"W{layer}"
        b_key = f"b{layer}"
        if w_key not in weights or b_key not in weights:
            break

        W = weights[w_key].astype(np.float32)
        b = weights[b_key].astype(np.float32)
        x = x @ W + b

        # For all but the last layer, use ReLU
        next_w_key = f"W{layer+1}"
        if next_w_key in weights:
            x = np.maximum(x, 0)
        layer += 1

    # If we never applied any layer, it's invalid
    if layer == 1:
        raise ValueError("Invalid .npz: missing W1/b1 weights.")

    # Map to image space
    x = _sigmoid(x)
    expected = img_size * img_size
    if x.shape[1] != expected:
        raise ValueError(
            f"Generator output shape mismatch: got {x.shape[1]} features, expected {expected} (= {img_size}×{img_size})."
        )
    return x.reshape(-1, img_size, img_size)


def make_example_mlp_weights(latent_dim, img_size, seed=123, hidden_sizes=(256, 512)):
    """Create random MLP weights that match the app's .npz inference format."""
    rng = np.random.RandomState(int(seed))
    sizes = [int(latent_dim), *map(int, hidden_sizes), int(img_size * img_size)]
    weights = {}
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        # He/Xavier-ish small init to avoid saturation
        scale = (
            np.sqrt(2.0 / max(fan_in, 1))
            if i < len(sizes) - 2
            else np.sqrt(1.0 / max(fan_in, 1))
        )
        W = (rng.randn(fan_in, fan_out) * scale).astype(np.float32)
        b = (rng.randn(fan_out) * (scale * 0.1)).astype(np.float32)
        weights[f"W{i+1}"] = W
        weights[f"b{i+1}"] = b
    return weights


def npz_bytes_from_weights(weights_dict):
    buf = BytesIO()
    np.savez(buf, **weights_dict)
    buf.seek(0)
    return buf


def images_to_zip_bytes(images, cmap, metadata, prefix="gan"):
    """Pack individual PNGs + metadata.json into a ZIP (BytesIO)."""
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # metadata
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        # images
        for i, img in enumerate(images, start=1):
            img_buf = BytesIO()
            plt.imsave(img_buf, img, cmap=cmap, format="png")
            zf.writestr(f"{prefix}_{i:04d}.png", img_buf.getvalue())

    zip_buf.seek(0)
    return zip_buf


def simulate_training(epochs, batch_size, lr_g, lr_d, seed=42):
    np.random.seed(seed)
    g_losses, d_losses, d_real_acc, d_fake_acc = [], [], [], []
    for e in range(epochs):
        t = e / max(epochs - 1, 1)
        noise = 0.08 * np.random.randn()
        d_loss = 1.386 * np.exp(-3.5 * t) + 0.35 + noise * (lr_d * 800)
        g_loss = 1.8 * np.exp(-2.8 * t) + 0.45 + noise * (lr_g * 800)
        d_real = min(0.52 + 0.45 * t + 0.02 * np.random.randn(), 0.98)
        d_fake = max(0.48 - 0.44 * t + 0.02 * np.random.randn(), 0.02)
        g_losses.append(max(g_loss, 0.2))
        d_losses.append(max(d_loss, 0.2))
        d_real_acc.append(float(np.clip(d_real, 0.01, 0.99)))
        d_fake_acc.append(float(np.clip(d_fake, 0.01, 0.99)))
    return g_losses, d_losses, d_real_acc, d_fake_acc


def make_plotly_layout(height=380, title=None):
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,11,30,0.7)",
        font=dict(color="#e2e8f0", family="Inter"),
        title=dict(text=title, font=dict(size=14, color="#a78bfa")) if title else None,
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(139,92,246,0.3)", borderwidth=1
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"
        ),
        margin=dict(t=50, b=40, l=50, r=20),
    )


def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center; padding:1.5rem 0 1rem;'>
      <div style='font-size:3.5rem; line-height:1;'>🧠</div>
      <div style='font-size:1.3rem; font-weight:800; color:#a78bfa; margin-top:.4rem;'>GAN Explorer</div>
      <div style='font-size:.78rem; color:#64748b; margin-top:.2rem; letter-spacing:.5px;'>
        INTERACTIVE DEEP LEARNING
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### ⚙️ Hyperparameters")
    latent_dim = st.slider(
        "🔲 Latent Dimension (z)",
        16,
        256,
        100,
        step=16,
        help="Size of the noise vector fed to the Generator",
    )
    epochs = st.slider(
        "🔄 Training Epochs",
        10,
        300,
        100,
        step=10,
        help="Number of full training iterations",
    )
    batch_size = st.selectbox(
        "📦 Batch Size",
        [16, 32, 64, 128, 256],
        index=2,
        help="Samples processed per gradient step",
    )
    lr_g = st.select_slider(
        "📈 Generator LR",
        options=[0.00005, 0.0001, 0.0002, 0.0005, 0.001],
        value=0.0002,
    )
    lr_d = st.select_slider(
        "📉 Discriminator LR",
        options=[0.00005, 0.0001, 0.0002, 0.0005, 0.001],
        value=0.0002,
    )

    st.divider()

    st.markdown("#### 🎨 Generation Settings")
    n_generate = st.slider("🖼️ Images to Generate", 4, 64, 16, step=4)
    img_size = st.selectbox(
        "📐 Image Resolution",
        [16, 28, 32, 64],
        index=1,
        help="Size of generated images in pixels",
    )
    color_scheme = st.selectbox(
        "🎨 Color Map",
        ["plasma", "viridis", "magma", "inferno", "cividis", "rainbow", "cool", "hot"],
        index=0,
    )
    random_seed = st.number_input(
        "🎲 Random Seed", min_value=0, max_value=9999, value=42, step=1
    )

    st.divider()

    st.markdown("#### 🔗 Connect with Me")
    st.markdown(
        """
    <a class='social-btn linkedin-btn'
       href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank'>
      <svg width='16' height='16' viewBox='0 0 24 24' fill='white'>
        <path d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/>
      </svg>
      LinkedIn
    </a>
    <a class='social-btn github-btn'
       href='https://github.com/HafsaIbrahim5' target='_blank'>
      <svg width='16' height='16' viewBox='0 0 24 24' fill='white'>
        <path d='M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z'/>
      </svg>
      GitHub
    </a>
    """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        """
    <div style='font-size:.75rem; color:#475569; text-align:center; line-height:1.6;'>
      Built with ❤️ by<br>
      <strong style='color:#a78bfa !important;'>Hafsa Ibrahim</strong><br>
      AI / ML Engineer
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════
st.markdown(
    """
<div class='hero'>
  <h1>🧠 GAN Explorer</h1>
  <p>Interactive Generative Adversarial Networks Dashboard</p>
  <span class='hero-badge'>⚡ Deep Learning · PyTorch · Generative AI</span>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════
# METRICS ROW
# ══════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
gen_params = latent_dim * 256 + 256 * 512 + 512 * 1024 + 1024 * img_size * img_size
disc_params = img_size * img_size * 1024 + 1024 * 512 + 512

for col, icon, label, val, sub in zip(
    [c1, c2, c3, c4, c5],
    ["🔲", "🔄", "📦", "🖼️", "⚡"],
    ["Latent Dim", "Epochs", "Batch Size", "Generate", "Total Params"],
    [latent_dim, epochs, batch_size, n_generate, f"{(gen_params+disc_params)//1000}K"],
    ["noise vector", "iterations", "per step", "images", "G + D combined"],
):
    with col:
        st.markdown(
            f"""
        <div class='metric-card'>
          <div class='metric-icon'>{icon}</div>
          <div class='metric-val'>{val}</div>
          <div style='font-size:.78rem; color:#a78bfa; font-weight:700; margin-top:.15rem;'>{label}</div>
          <div class='metric-lab'>{sub}</div>
        </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════
tabs = st.tabs(
    [
        "📖 About GANs",
        "🎨 Image Generator",
        "📈 Training Simulator",
        "🏗️ Architecture",
        "📊 Analytics",
        "🔬 Latent Space",
        "🆚 GAN Variants",
        "💡 Tips & Tricks",
    ]
)

# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 1 — ABOUT GANs                                      ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[0]:
    st.markdown(
        "<div class='section-title'>What Are Generative Adversarial Networks?</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(
            """
        <div class='card'>
          <h4 style='color:#a78bfa; margin-top:0;'>🧬 Core Concept</h4>
          <p>GANs, introduced by <strong>Ian Goodfellow et al. in 2014</strong>, consist of two neural
          networks locked in a minimax game:</p>
          <ul>
            <li><strong style='color:#67e8f9;'>Generator (G)</strong> — Takes random noise <em>z</em>
                from a latent space and maps it to realistic-looking data</li>
            <li><strong style='color:#f9a8d4;'>Discriminator (D)</strong> — Learns to distinguish
                real data from Generator fakes, outputting a probability</li>
          </ul>
          <p>Through competition, <strong>G</strong> learns to fool <strong>D</strong>, while <strong>D</strong>
          becomes a better critic. At convergence, G produces data indistinguishable from real samples.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class='card'>
          <h4 style='color:#a78bfa; margin-top:0;'>📐 Mathematical Objective</h4>
          <div style='background:rgba(0,0,0,.35); padding:1.1rem; border-radius:10px;
                      font-family:"JetBrains Mono",monospace; font-size:.9rem; line-height:2;
                      color:#e2e8f0; border: 1px solid rgba(139,92,246,0.2);'>
            <span style='color:#a78bfa;'>min</span><sub>G</sub>
            <span style='color:#f472b6;'> max</span><sub>D</sub>
            V(D,G) =<br>
            &nbsp;&nbsp;<span style='color:#67e8f9;'>𝔼</span><sub>x~p_data</sub>[log D(x)]<br>
            &nbsp;+ <span style='color:#67e8f9;'>𝔼</span><sub>z~p_z</sub>[log(1 − D(G(z)))]
          </div>
          <p style='margin-top:.8rem; font-size:.88rem; color:#94a3b8;'>
            D maximizes this value (better critic); G minimizes it (better generator).
            Optimal solution: G(z) ~ p_data, D(x) = ½ everywhere.
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            """
        <div class='card'>
          <h4 style='color:#a78bfa; margin-top:0;'>🚀 Key Applications</h4>
          <div style='display:grid; grid-template-columns:1fr 1fr; gap:.5rem; font-size:.88rem;'>
            <div style='background:rgba(102,126,234,.1); padding:.5rem; border-radius:8px;'>🖼️ Image Synthesis</div>
            <div style='background:rgba(118,75,162,.1); padding:.5rem; border-radius:8px;'>🎭 Face Generation</div>
            <div style='background:rgba(168,85,247,.1); padding:.5rem; border-radius:8px;'>🎨 Style Transfer</div>
            <div style='background:rgba(236,72,153,.1); padding:.5rem; border-radius:8px;'>💊 Drug Discovery</div>
            <div style='background:rgba(245,158,11,.1); padding:.5rem; border-radius:8px;'>🔊 Audio Synthesis</div>
            <div style='background:rgba(16,185,129,.1); padding:.5rem; border-radius:8px;'>📹 Video Prediction</div>
            <div style='background:rgba(6,182,212,.1); padding:.5rem; border-radius:8px;'>🧬 Medical Imaging</div>
            <div style='background:rgba(99,102,241,.1); padding:.5rem; border-radius:8px;'>🗺️ Map Generation</div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class='card'>
          <h4 style='color:#a78bfa; margin-top:0;'>📅 GAN Timeline</h4>
          <div style='font-size:.85rem; line-height:2;'>
            <div><span style='color:#667eea; font-weight:700;'>2014</span>
                 <span style='color:#94a3b8;'> — Original GAN (Goodfellow et al.)</span></div>
            <div><span style='color:#764ba2; font-weight:700;'>2015</span>
                 <span style='color:#94a3b8;'> — DCGAN: Deep Convolutional GAN</span></div>
            <div><span style='color:#a855f7; font-weight:700;'>2017</span>
                 <span style='color:#94a3b8;'> — WGAN: Wasserstein stability</span></div>
            <div><span style='color:#ec4899; font-weight:700;'>2018</span>
                 <span style='color:#94a3b8;'> — BigGAN: Large-scale synthesis</span></div>
            <div><span style='color:#f59e0b; font-weight:700;'>2019</span>
                 <span style='color:#94a3b8;'> — StyleGAN: Photorealistic faces</span></div>
            <div><span style='color:#10b981; font-weight:700;'>2021</span>
                 <span style='color:#94a3b8;'> — StyleGAN3: Alias-free</span></div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Training process
    st.markdown(
        "<div class='section-title' style='margin-top:1.5rem;'>🔄 Training Process — Step by Step</div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("1", "Sample Real Data", "Draw batch x from p_data(x)", "#667eea"),
        ("2", "Sample Noise", "z ~ N(0,I) from latent space ℝᵈ", "#764ba2"),
        ("3", "Generator Forward", "G(z) → fake images via deep network", "#a855f7"),
        (
            "4",
            "Discriminator Step",
            "D classifies real vs fake; compute BCE loss",
            "#ec4899",
        ),
        ("5", "Update D", "Maximize log D(x) + log(1−D(G(z)))", "#f59e0b"),
        ("6", "Update G", "Minimize log(1−D(G(z))) → fool Discriminator", "#10b981"),
    ]
    for i in range(0, len(steps), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(steps):
                num, title, desc, color = steps[i + j]
                with col:
                    st.markdown(
                        f"""
                    <div class='card' style='border-left:4px solid {color};'>
                      <div class='step-badge'>{num}</div>
                      <div style='font-weight:700; color:#e2e8f0; margin:.3rem 0;'>{title}</div>
                      <div style='font-size:.84rem; color:#94a3b8;'>{desc}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

    # Mode collapse & challenges
    st.markdown(
        "<div class='section-title' style='margin-top:1.5rem;'>⚠️ Common GAN Challenges</div>",
        unsafe_allow_html=True,
    )
    ch1, ch2, ch3 = st.columns(3)
    challenges = [
        (
            "🌀",
            "Mode Collapse",
            "G collapses to a single output, ignoring diversity. Fix: Minibatch discrimination, Unrolled GANs.",
            "#ef4444",
        ),
        (
            "💥",
            "Training Instability",
            "Loss oscillates or diverges. Fix: WGAN-GP, spectral normalization, careful LR tuning.",
            "#f59e0b",
        ),
        (
            "🔍",
            "Evaluation Difficulty",
            "No direct loss metric = model quality. Fix: FID score, Inception Score, human evaluation.",
            "#06b6d4",
        ),
    ]
    for col, (icon, title, desc, color) in zip([ch1, ch2, ch3], challenges):
        with col:
            st.markdown(
                f"""
            <div class='card' style='border-color:{color}44;'>
              <div style='font-size:1.8rem;'>{icon}</div>
              <div style='font-weight:700; color:{color}; margin:.3rem 0;'>{title}</div>
              <div style='font-size:.84rem; color:#94a3b8;'>{desc}</div>
            </div>""",
                unsafe_allow_html=True,
            )


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 2 — IMAGE GENERATOR                                 ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[1]:
    st.markdown(
        "<div class='section-title'>🎨 GAN Image Generator</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class='card'>
      Sample random latent vectors <strong>z ~ N(0,I)</strong> and feed them through a simulated
      Generator network to produce synthetic images. Each generation uses a different random seed
      to demonstrate the diversity of the latent space.
    </div>""",
        unsafe_allow_html=True,
    )

    col_b1, col_b2, col_b3, col_b4 = st.columns([1, 1, 1, 2])
    with col_b1:
        do_generate = st.button("🎲 Generate New Batch", use_container_width=True)
    with col_b2:
        do_interpolate = st.button("🌀 Interpolate z₁→z₂", use_container_width=True)
    with col_b3:
        do_grid = st.button("🔲 Latent Grid (4×4)", use_container_width=True)
    with col_b4:
        show_noise = st.checkbox("Show noise heatmap", value=False)

    st.markdown("#### ⚡ Inference Mode (No Training)")
    st.markdown(
        "<div class='card' style='font-size:.88rem; color:#94a3b8;'>"
        "<strong>Important:</strong> The uploader below expects a <strong>.npz weights file</strong> "
        "(a pretrained Generator), not images. If you don't have one, download the example weights "
        "button and try it instantly."
        "</div>",
        unsafe_allow_html=True,
    )
    gen_mode = st.radio(
        "Choose generator backend",
        ["Simulated (built-in)", "Pretrained NumPy weights (.npz)"],
        horizontal=True,
        help="This app won't train models. You can either use the built-in simulated generator or upload a pretrained generator exported as a .npz file.",
    )

    weights_npz = None
    weights_name = None
    if gen_mode == "Pretrained NumPy weights (.npz)":
        ex_weights = make_example_mlp_weights(
            latent_dim, img_size, seed=int(random_seed)
        )
        ex_npz = npz_bytes_from_weights(ex_weights)
        st.download_button(
            "⬇️ Download example weights (.npz)",
            ex_npz,
            file_name=f"example_generator_ld{latent_dim}_s{img_size}.npz",
            mime="application/octet-stream",
            help="Example random weights that match this app's expected .npz format (W1,b1,W2,b2,...).",
            use_container_width=False,
        )
        up = st.file_uploader(
            "Upload generator weights (.npz)",
            type=["npz"],
            help="Expected keys: W1,b1,W2,b2,... Output must be img_size×img_size per sample.",
        )
        if up is not None:
            try:
                weights_npz = dict(np.load(up))
                weights_name = up.name
                st.success(f"Loaded weights: {weights_name}")
            except Exception as e:
                st.error(f"Failed to load .npz weights: {e}")
                weights_npz = None

    # In pretrained mode: don't show any images until weights are uploaded
    if gen_mode == "Pretrained NumPy weights (.npz)" and weights_npz is None:
        for k in [
            "gen_imgs",
            "gen_noise",
            "gen_seed",
            "gen_backend",
            "gen_weights_name",
        ]:
            if k in st.session_state:
                del st.session_state[k]

        st.info(
            "Upload a `.npz` weights file, then click **Generate New Batch** to see images."
        )
        st.stop()

    # Generate
    if do_generate or "gen_imgs" not in st.session_state:
        noise = generate_noise(n_generate, latent_dim, seed=int(random_seed))
        try:
            if weights_npz is not None:
                st.session_state.gen_imgs = mlp_generator_from_npz(
                    noise, weights_npz, img_size
                )
                st.session_state.gen_backend = "npz"
                st.session_state.gen_weights_name = weights_name
            else:
                st.session_state.gen_imgs = simple_generator(noise, img_size)
                st.session_state.gen_backend = "sim"
                st.session_state.gen_weights_name = None
        except Exception as e:
            st.warning(f"Falling back to simulated generator (weights error): {e}")
            st.session_state.gen_imgs = simple_generator(noise, img_size)
            st.session_state.gen_backend = "sim"
            st.session_state.gen_weights_name = None
        st.session_state.gen_noise = noise
        st.session_state.gen_seed = random_seed

    imgs = st.session_state.gen_imgs
    ncols = min(8, n_generate)
    nrows = (n_generate + ncols - 1) // ncols

    cmaps_cycle = [color_scheme, "plasma", "viridis", "magma", "inferno", "cividis"]

    fig_gen, axes_gen = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6))
    fig_gen.patch.set_facecolor("#0d0b1e")
    axes_flat = np.array(axes_gen).flatten() if n_generate > 1 else [axes_gen]
    for i, ax in enumerate(axes_flat):
        if i < len(imgs):
            ax.imshow(
                imgs[i],
                cmap=cmaps_cycle[i % len(cmaps_cycle)],
                interpolation="bilinear",
            )
            ax.set_title(
                f"#{i+1}", fontsize=7, color="#a78bfa", pad=2, fontweight="bold"
            )
        ax.axis("off")
    fig_gen.tight_layout(pad=0.3)

    st.pyplot(fig_gen, use_container_width=True)

    # Download button
    buf = fig_to_bytes(fig_gen)
    st.download_button(
        "⬇️ Download Generated Images",
        buf,
        "gan_generated.png",
        "image/png",
        use_container_width=False,
    )

    # Download as ZIP (individual PNGs + metadata)
    zip_meta = {
        "seed": int(st.session_state.get("gen_seed", random_seed)),
        "latent_dim": int(latent_dim),
        "img_size": int(img_size),
        "n_images": int(len(imgs)),
        "backend": st.session_state.get("gen_backend", "sim"),
        "weights_file": st.session_state.get("gen_weights_name"),
    }
    zip_buf = images_to_zip_bytes(
        imgs,
        cmap=color_scheme,
        metadata=zip_meta,
        prefix="gan_image",
    )
    st.download_button(
        "📦 Download ZIP (images + metadata)",
        zip_buf,
        file_name="gan_images.zip",
        mime="application/zip",
        use_container_width=False,
    )
    plt.close(fig_gen)

    # Interpolation
    if do_interpolate:
        st.markdown("#### 🌀 Latent Space Interpolation")
        st.markdown(
            """
        <div class='card' style='font-size:.88rem; color:#94a3b8;'>
          Smooth interpolation between two random points <strong>z₁</strong> and <strong>z₂</strong>
          in the latent space. Notice how the images morph continuously — evidence that the latent
          space is smooth and continuous.
        </div>""",
            unsafe_allow_html=True,
        )

        z1 = generate_noise(1, latent_dim, seed=int(random_seed))[0]
        z2 = generate_noise(1, latent_dim, seed=int(random_seed) + 999)[0]
        steps_interp = 12
        alphas = np.linspace(0, 1, steps_interp)
        interp_imgs = [
            simple_generator(((1 - a) * z1 + a * z2)[np.newaxis], img_size)[0]
            for a in alphas
        ]

        fig_int, axes_int = plt.subplots(
            1, steps_interp, figsize=(steps_interp * 1.5, 1.8)
        )
        fig_int.patch.set_facecolor("#0d0b1e")
        for i, (ax, img) in enumerate(zip(axes_int, interp_imgs)):
            ax.imshow(img, cmap=color_scheme)
            ax.set_title(f"{alphas[i]:.2f}", fontsize=7, color="#a78bfa", pad=1)
            ax.axis("off")
        plt.suptitle(
            "Latent Interpolation  z₁ ──────────────────── z₂",
            color="white",
            fontsize=9,
            y=1.02,
        )
        fig_int.tight_layout(pad=0.2)
        st.pyplot(fig_int, use_container_width=True)
        plt.close(fig_int)

    # 4×4 Latent Grid
    if do_grid:
        st.markdown("#### 🔲 Latent Space Grid Exploration")
        st.markdown(
            """
        <div class='card' style='font-size:.88rem; color:#94a3b8;'>
          Explore a 4×4 grid in the latent space by varying two latent dimensions.
          Rows vary z[0], columns vary z[1] — revealing structured patterns.
        </div>""",
            unsafe_allow_html=True,
        )
        grid_n = 4
        base_z = generate_noise(1, latent_dim, seed=int(random_seed))[0].copy()
        vals = np.linspace(-2, 2, grid_n)

        fig_grid, ax_grid = plt.subplots(
            grid_n, grid_n, figsize=(grid_n * 1.6, grid_n * 1.6)
        )
        fig_grid.patch.set_facecolor("#0d0b1e")
        for ri, v0 in enumerate(vals):
            for ci, v1 in enumerate(vals):
                z = base_z.copy()
                z[0] = v0
                z[1] = v1
                img = simple_generator(z[np.newaxis], img_size)[0]
                ax_grid[ri][ci].imshow(img, cmap=color_scheme)
                ax_grid[ri][ci].axis("off")
        plt.suptitle(
            "Latent Grid: z[0] (rows) × z[1] (cols)", color="white", fontsize=9
        )
        fig_grid.tight_layout(pad=0.2)
        st.pyplot(fig_grid, use_container_width=True)
        plt.close(fig_grid)

    # Noise heatmap
    if show_noise:
        with st.expander("🔍 Latent Noise Vectors Heatmap", expanded=True):
            noise_slice = st.session_state.gen_noise[
                : min(8, n_generate), : min(64, latent_dim)
            ]
            fig_hm = px.imshow(
                noise_slice,
                labels=dict(x="Latent Dimension", y="Sample Index", color="Value"),
                color_continuous_scale="rdylbu",  # corrected colorscale name
                title="Sampled Noise Vectors z (first 64 dims)",
                aspect="auto",
            )
            fig_hm.update_layout(**make_plotly_layout(300))
            st.plotly_chart(fig_hm, use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 3 — TRAINING SIMULATOR                              ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[2]:
    st.markdown(
        "<div class='section-title'>📈 Training Simulator</div>", unsafe_allow_html=True
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 3])
    with col_ctrl1:
        run_sim = st.button("▶️ Run Simulation", use_container_width=True)
    with col_ctrl2:
        animate = st.checkbox(
            "⚡ Animate", value=False, help="Watch training unfold in real time"
        )
    with col_ctrl3:
        st.markdown(
            """
        <div style='padding:.4rem; font-size:.85rem; color:#94a3b8;'>
          Simulates a GAN training loop using realistic loss curves.
          Adjust hyperparameters in the sidebar to see their effect.
        </div>""",
            unsafe_allow_html=True,
        )

    if run_sim or "g_losses" not in st.session_state:
        if animate:
            prog = st.progress(0, "🏋️ Training in progress...")
            chart_ph = st.empty()
            g_l_anim, d_l_anim = [], []
            gl_all, dl_all, ra_all, fa_all = simulate_training(
                epochs, batch_size, lr_g, lr_d, int(random_seed)
            )
            for ep in range(epochs):
                g_l_anim.append(gl_all[ep])
                d_l_anim.append(dl_all[ep])
                prog.progress((ep + 1) / epochs, f"Epoch {ep+1} / {epochs}")
                if ep % max(1, epochs // 25) == 0:
                    tmp = go.Figure()
                    tmp.add_trace(
                        go.Scatter(
                            y=g_l_anim,
                            name="Generator Loss",
                            line=dict(color="#a78bfa", width=2),
                        )
                    )
                    tmp.add_trace(
                        go.Scatter(
                            y=d_l_anim,
                            name="Discriminator Loss",
                            line=dict(color="#f472b6", width=2),
                        )
                    )
                    tmp.update_layout(**make_plotly_layout(280, "Live Training Loss"))
                    chart_ph.plotly_chart(tmp, use_container_width=True)
                    time.sleep(0.03)
            prog.empty()
            chart_ph.empty()
        else:
            gl_all, dl_all, ra_all, fa_all = simulate_training(
                epochs, batch_size, lr_g, lr_d, int(random_seed)
            )

        st.session_state.g_losses = gl_all
        st.session_state.d_losses = dl_all
        st.session_state.d_real = ra_all
        st.session_state.d_fake = fa_all

    g_l = st.session_state.g_losses
    d_l = st.session_state.d_losses
    d_r = st.session_state.d_real
    d_f = st.session_state.d_fake
    ep_x = list(range(1, len(g_l) + 1))

    # Loss + Accuracy charts
    fig_train = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Generator vs Discriminator Loss",
            "Discriminator Accuracy",
            "G/D Loss Ratio",
            "Cumulative Loss",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    # Loss curves
    fig_train.add_trace(
        go.Scatter(x=ep_x, y=g_l, name="G Loss", line=dict(color="#a78bfa", width=2.5)),
        row=1,
        col=1,
    )
    fig_train.add_trace(
        go.Scatter(x=ep_x, y=d_l, name="D Loss", line=dict(color="#f472b6", width=2.5)),
        row=1,
        col=1,
    )
    # Accuracy
    fig_train.add_trace(
        go.Scatter(
            x=ep_x,
            y=d_r,
            name="D(real)",
            line=dict(color="#34d399", width=2),
            fill="tonexty",
            fillcolor="rgba(52,211,153,0.05)",
        ),
        row=1,
        col=2,
    )
    fig_train.add_trace(
        go.Scatter(x=ep_x, y=d_f, name="D(fake)", line=dict(color="#fbbf24", width=2)),
        row=1,
        col=2,
    )
    fig_train.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#475569",
        annotation_text="Nash Eq. (0.5)",
        row=1,
        col=2,
    )
    # Ratio
    ratio = [g / d if d > 0 else 1 for g, d in zip(g_l, d_l)]
    fig_train.add_trace(
        go.Scatter(
            x=ep_x, y=ratio, name="G/D Ratio", line=dict(color="#67e8f9", width=2)
        ),
        row=2,
        col=1,
    )
    fig_train.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="#475569",
        annotation_text="Balanced",
        row=2,
        col=1,
    )
    # Cumulative
    cum_g = np.cumsum(g_l).tolist()
    cum_d = np.cumsum(d_l).tolist()
    fig_train.add_trace(
        go.Scatter(
            x=ep_x,
            y=cum_g,
            name="Cum G Loss",
            line=dict(color="#a78bfa", width=1.5, dash="dot"),
        ),
        row=2,
        col=2,
    )
    fig_train.add_trace(
        go.Scatter(
            x=ep_x,
            y=cum_d,
            name="Cum D Loss",
            line=dict(color="#f472b6", width=1.5, dash="dot"),
        ),
        row=2,
        col=2,
    )

    fig_train.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,11,30,0.7)",
        font_color="#e2e8f0",
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(139,92,246,0.3)",
            borderwidth=1,
        ),
        margin=dict(t=50, b=30, l=50, r=20),
    )
    for ax in [
        "xaxis",
        "xaxis2",
        "xaxis3",
        "xaxis4",
        "yaxis",
        "yaxis2",
        "yaxis3",
        "yaxis4",
    ]:
        fig_train.update_layout(**{ax: dict(gridcolor="rgba(255,255,255,0.05)")})

    st.plotly_chart(fig_train, use_container_width=True)

    # Summary metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, label, val, color in zip(
        [m1, m2, m3, m4, m5, m6],
        [
            "Final G Loss",
            "Final D Loss",
            "D(real) Acc",
            "D(fake) Acc",
            "Min G Loss",
            "Balance",
        ],
        [
            f"{g_l[-1]:.3f}",
            f"{d_l[-1]:.3f}",
            f"{d_r[-1]*100:.1f}%",
            f"{(1-d_f[-1])*100:.1f}%",
            f"{min(g_l):.3f}",
            "✅ Good" if abs(g_l[-1] - d_l[-1]) < 0.3 else "⚠️ Check",
        ],
        ["#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#67e8f9", "#10b981"],
    ):
        with col:
            st.markdown(
                f"""
            <div class='metric-card' style='border-color:{color}44; padding:.9rem .6rem;'>
              <div class='metric-val' style='color:{color}; font-size:1.4rem;'>{val}</div>
              <div class='metric-lab'>{label}</div>
            </div>""",
                unsafe_allow_html=True,
            )


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 4 — ARCHITECTURE                                    ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[3]:
    st.markdown(
        "<div class='section-title'>🏗️ Network Architecture</div>",
        unsafe_allow_html=True,
    )

    col_g, col_d = st.columns(2)
    g_p1 = latent_dim * 256
    g_p2 = 256 * 512
    g_p3 = 512 * 1024
    g_p4 = 1024 * img_size * img_size
    d_p1 = img_size * img_size * 1024
    d_p2 = 1024 * 512
    d_p3 = 512 * 1

    with col_g:
        st.markdown(
            f"""
        <div class='card' style='border-color:#a78bfa44;'>
          <h4 style='color:#a78bfa; margin-top:0;'>⬆️ Generator  G(z)</h4>
          <p style='font-size:.82rem; color:#94a3b8;'>Noise → Fake Image</p>
          <div style='font-family:"JetBrains Mono",monospace; font-size:.82rem; line-height:2.1;'>
            <div style='color:#67e8f9; font-weight:600;'>Input: z ∈ ℝ<sup>{latent_dim}</sup></div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;↓ Linear({latent_dim}→256) &nbsp;[{g_p1:,} params]</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;&nbsp;&nbsp;BatchNorm1d(256) + LeakyReLU(0.2)</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;↓ Linear(256→512) &nbsp;&nbsp;[{g_p2:,} params]</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;&nbsp;&nbsp;BatchNorm1d(512) + LeakyReLU(0.2)</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;↓ Linear(512→1024) &nbsp;[{g_p3:,} params]</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;&nbsp;&nbsp;BatchNorm1d(1024) + LeakyReLU(0.2)</div>
            <div style='color:#a78bfa;'>&nbsp;&nbsp;↓ Linear(1024→{img_size*img_size}) [{g_p4:,} params]</div>
            <div style='color:#a78bfa;'>&nbsp;&nbsp;&nbsp;&nbsp;Tanh() → range [-1, 1]</div>
            <div style='color:#f9a8d4; font-weight:600;'>Output: img ∈ ℝ<sup>{img_size}×{img_size}</sup></div>
          </div>
          <div style='margin-top:1rem; background:rgba(167,139,250,.12); border-radius:10px; padding:.8rem;'>
            <strong style='color:#a78bfa;'>Total: {g_p1+g_p2+g_p3+g_p4:,} parameters</strong>
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col_d:
        st.markdown(
            f"""
        <div class='card' style='border-color:#f472b644;'>
          <h4 style='color:#f472b6; margin-top:0;'>⬇️ Discriminator  D(x)</h4>
          <p style='font-size:.82rem; color:#94a3b8;'>Image → Real/Fake Probability</p>
          <div style='font-family:"JetBrains Mono",monospace; font-size:.82rem; line-height:2.1;'>
            <div style='color:#67e8f9; font-weight:600;'>Input: x ∈ ℝ<sup>{img_size}×{img_size}</sup></div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;↓ Linear({img_size*img_size}→1024) [{d_p1:,} params]</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;&nbsp;&nbsp;LeakyReLU(0.2) + Dropout(0.3)</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;↓ Linear(1024→512) &nbsp;[{d_p2:,} params]</div>
            <div style='color:#94a3b8;'>&nbsp;&nbsp;&nbsp;&nbsp;LeakyReLU(0.2) + Dropout(0.3)</div>
            <div style='color:#f472b6;'>&nbsp;&nbsp;↓ Linear(512→1) &nbsp;&nbsp;&nbsp;[{d_p3:,} params]</div>
            <div style='color:#f472b6;'>&nbsp;&nbsp;&nbsp;&nbsp;Sigmoid() → p(real)</div>
            <div style='color:#f9a8d4; font-weight:600;'>Output: p ∈ [0, 1]</div>
          </div>
          <div style='margin-top:1rem; background:rgba(244,114,182,.12); border-radius:10px; padding:.8rem;'>
            <strong style='color:#f472b6;'>Total: {d_p1+d_p2+d_p3:,} parameters</strong>
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

    # Param visualizations
    fig_arch = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "bar"}]],
        subplot_titles=[
            "G vs D Parameter Share",
            "Generator Layer Breakdown",
            "Discriminator Layer Breakdown",
        ],
        horizontal_spacing=0.08,
    )
    fig_arch.add_trace(
        go.Pie(
            labels=["Generator", "Discriminator"],
            values=[g_p1 + g_p2 + g_p3 + g_p4, d_p1 + d_p2 + d_p3],
            marker=dict(
                colors=["#a78bfa", "#f472b6"], line=dict(color="#0d0b1e", width=2)
            ),
            hole=0.45,
            textfont=dict(color="white"),
        ),
        row=1,
        col=1,
    )

    fig_arch.add_trace(
        go.Bar(
            x=[f"z→256", "256→512", "512→1024", f"1024→{img_size*img_size}"],
            y=[g_p1, g_p2, g_p3, g_p4],
            marker=dict(
                color=["#667eea", "#764ba2", "#a855f7", "#ec4899"],
                line=dict(color="rgba(255,255,255,0.1)", width=1),
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig_arch.add_trace(
        go.Bar(
            x=[f"{img_size*img_size}→1024", "1024→512", "512→1"],
            y=[d_p1, d_p2, d_p3],
            marker=dict(
                color=["#f472b6", "#ec4899", "#be185d"],
                line=dict(color="rgba(255,255,255,0.1)", width=1),
            ),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig_arch.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,11,30,0.7)",
        font_color="#e2e8f0",
        margin=dict(t=50, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_arch, use_container_width=True)

    # PyTorch code preview
    with st.expander("📋 View PyTorch Implementation Code"):
        st.code(
            f"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim={latent_dim}, img_size={img_size}):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z).view(-1, 1, {img_size}, {img_size})


class Discriminator(nn.Module):
    def __init__(self, img_size={img_size}):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.model(flat)


# Training loop
def train_gan(G, D, dataloader, epochs={epochs}, lr_g={lr_g}, lr_d={lr_d}):
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for real_imgs, _ in dataloader:
            batch = real_imgs.size(0)

            # ── Train Discriminator ──
            z = torch.randn(batch, {latent_dim})
            fake = G(z).detach()
            loss_D = (criterion(D(real_imgs.view(batch, -1)),
                                torch.ones(batch, 1))
                    + criterion(D(fake.view(batch, -1)),
                                torch.zeros(batch, 1))) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ── Train Generator ──
            z = torch.randn(batch, {latent_dim})
            fake = G(z)
            loss_G = criterion(D(fake.view(batch, -1)), torch.ones(batch, 1))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
""",
            language="python",
        )


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 5 — ANALYTICS DASHBOARD                             ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[4]:
    st.markdown(
        "<div class='section-title'>📊 Analytics Dashboard</div>",
        unsafe_allow_html=True,
    )

    if "g_losses" not in st.session_state:
        st.info(
            "▶️ Run the **Training Simulator** (Tab 3) first to unlock the Analytics Dashboard!"
        )
    else:
        g_l = st.session_state.g_losses
        d_l = st.session_state.d_losses
        ep_x = list(range(1, len(g_l) + 1))

        # ── Hyperparameter Sensitivity Heatmap ──
        st.markdown("#### 🌡️ Hyperparameter Sensitivity: Final Generator Loss")
        lrs = [0.00005, 0.0001, 0.0002, 0.0005, 0.001]
        heat_matrix = []
        for lr_gi in lrs:
            row = []
            for lr_di in lrs:
                gl, _, _, _ = simulate_training(epochs, batch_size, lr_gi, lr_di, 42)
                row.append(round(gl[-1], 3))
            heat_matrix.append(row)

        fig_heat = px.imshow(
            heat_matrix,
            x=[str(x) for x in lrs],
            y=[str(x) for x in lrs],
            labels=dict(x="LR Discriminator", y="LR Generator", color="Final G Loss"),
            color_continuous_scale="rdylbu",  # corrected colorscale name
            text_auto=True,
        )
        fig_heat.update_layout(
            **make_plotly_layout(360, "Final G Loss across LR Combinations")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Distribution + Smoothed ──
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=g_l,
                    name="G Loss",
                    marker_color="#a78bfa",
                    opacity=0.75,
                    nbinsx=20,
                )
            )
            fig_hist.add_trace(
                go.Histogram(
                    x=d_l,
                    name="D Loss",
                    marker_color="#f472b6",
                    opacity=0.75,
                    nbinsx=20,
                )
            )
            fig_hist.update_layout(
                barmode="overlay", **make_plotly_layout(300, "Loss Distribution")
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_a2:
            smooth = lambda a, w=8: np.convolve(
                a, np.ones(w) / w, mode="valid"
            ).tolist()
            sg = smooth(g_l)
            sd = smooth(d_l)
            sx = list(range(1, len(sg) + 1))
            fig_sm = go.Figure()
            fig_sm.add_trace(
                go.Scatter(
                    x=sx, y=sg, name="G (smooth)", line=dict(color="#a78bfa", width=3)
                )
            )
            fig_sm.add_trace(
                go.Scatter(
                    x=sx, y=sd, name="D (smooth)", line=dict(color="#f472b6", width=3)
                )
            )
            fig_sm.add_trace(
                go.Scatter(
                    x=ep_x,
                    y=g_l,
                    name="G (raw)",
                    line=dict(color="#a78bfa", width=1, dash="dot"),
                    opacity=0.35,
                )
            )
            fig_sm.update_layout(**make_plotly_layout(300, "Smoothed vs Raw Loss"))
            st.plotly_chart(fig_sm, use_container_width=True)

        # ── Batch size comparison ──
        st.markdown("#### 📦 Batch Size Impact on Training Stability")
        fig_bs = go.Figure()
        colors_bs = ["#667eea", "#a78bfa", "#f472b6", "#fbbf24"]
        for bs, color in zip([16, 32, 64, 128], colors_bs):
            gl_bs, _, _, _ = simulate_training(epochs, bs, lr_g, lr_d, 42)
            fig_bs.add_trace(
                go.Scatter(y=gl_bs, name=f"BS={bs}", line=dict(color=color, width=2))
            )
        fig_bs.update_layout(**make_plotly_layout(320, "Generator Loss by Batch Size"))
        st.plotly_chart(fig_bs, use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 6 — LATENT SPACE EXPLORER                           ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[5]:
    st.markdown(
        "<div class='section-title'>🔬 Latent Space Explorer</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class='card'>
      The <strong>latent space</strong> z ∈ ℝᵈ is where the Generator lives.
      Understanding its geometry helps you control and navigate the generative process.
      Below you can visualize noise distributions, t-SNE projections, and spherical interpolation.
    </div>""",
        unsafe_allow_html=True,
    )

    col_ls1, col_ls2 = st.columns(2)

    with col_ls1:
        # Noise distribution
        n_samples = 500
        z_samples = np.random.randn(n_samples, latent_dim)
        fig_dist = go.Figure()
        for dim in range(min(5, latent_dim)):
            fig_dist.add_trace(
                go.Violin(
                    y=z_samples[:, dim],
                    name=f"z[{dim}]",
                    box_visible=True,
                    meanline_visible=True,
                    line_color=["#667eea", "#a78bfa", "#f472b6", "#fbbf24", "#34d399"][
                        dim
                    ],
                )
            )
        fig_dist.update_layout(
            **make_plotly_layout(350, "Latent Dimensions Distribution (z ~ N(0,I))")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_ls2:
        # 2D PCA-like scatter of latent vectors
        np.random.seed(int(random_seed))
        z_2d = np.random.randn(200, latent_dim)
        # Simulate 2D projection (random projection for demo)
        proj = np.random.randn(latent_dim, 2) / np.sqrt(latent_dim)
        coords = z_2d @ proj
        colors_scatter = np.linalg.norm(z_2d, axis=1)  # color by L2 norm

        fig_scatter = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=colors_scatter,
            color_continuous_scale="plasma",
            labels=dict(x="Component 1", y="Component 2", color="||z||"),
            title="2D Projection of Latent Vectors",
        )
        fig_scatter.update_traces(marker=dict(size=5, opacity=0.7))
        fig_scatter.update_layout(**make_plotly_layout(350))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Spherical interpolation
    st.markdown("#### 🌐 Spherical Interpolation (Slerp)")
    st.markdown(
        """
    <div class='card' style='font-size:.88rem; color:#94a3b8;'>
      <strong>Slerp</strong> (Spherical Linear Interpolation) interpolates on the surface of a
      hypersphere, producing smoother transitions than linear interpolation in high-dimensional
      latent spaces. Used in practice by StyleGAN and other state-of-the-art models.
    </div>""",
        unsafe_allow_html=True,
    )

    slerp_steps = st.slider("Interpolation Steps", 6, 20, 10, step=2)
    z1 = generate_noise(1, latent_dim, seed=int(random_seed))[0]
    z2 = generate_noise(1, latent_dim, seed=int(random_seed) + 77)[0]

    def slerp(z1, z2, t):
        norm1 = np.linalg.norm(z1)
        norm2 = np.linalg.norm(z2)
        z1n = z1 / norm1
        z2n = z2 / norm2
        dot = np.clip(np.dot(z1n, z2n), -1, 1)
        omega = np.arccos(dot)
        if abs(omega) < 1e-6:
            return (1 - t) * z1 + t * z2
        return (np.sin((1 - t) * omega) / np.sin(omega)) * z1 + (
            np.sin(t * omega) / np.sin(omega)
        ) * z2

    alphas_s = np.linspace(0, 1, slerp_steps)
    slerp_imgs = [
        simple_generator(slerp(z1, z2, a)[np.newaxis], img_size)[0] for a in alphas_s
    ]

    fig_slerp, axes_s = plt.subplots(1, slerp_steps, figsize=(slerp_steps * 1.5, 1.8))
    fig_slerp.patch.set_facecolor("#0d0b1e")
    for i, (ax, img) in enumerate(zip(axes_s, slerp_imgs)):
        ax.imshow(img, cmap=color_scheme)
        ax.set_title(f"{alphas_s[i]:.2f}", fontsize=7, color="#a78bfa", pad=1)
        ax.axis("off")
    plt.suptitle("Slerp Interpolation  z₁ ─── z₂", color="white", fontsize=9, y=1.04)
    fig_slerp.tight_layout(pad=0.2)
    st.pyplot(fig_slerp, use_container_width=True)
    plt.close(fig_slerp)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 7 — GAN VARIANTS                                    ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[6]:
    st.markdown(
        "<div class='section-title'>🆚 GAN Variants Comparison</div>",
        unsafe_allow_html=True,
    )

    variants = [
        {
            "name": "Vanilla GAN",
            "year": 2014,
            "loss": "Binary Cross-Entropy",
            "pros": "Simple, foundational, easy to implement",
            "cons": "Mode collapse, training instability, vanishing gradients",
            "use": "Learning & education, simple datasets",
            "color": "#667eea",
            "icon": "🧱",
        },
        {
            "name": "DCGAN",
            "year": 2015,
            "loss": "BCE + Conv layers",
            "pros": "Stable training, good image quality via CNNs",
            "cons": "Limited resolution, still prone to mode collapse",
            "use": "Image generation, face synthesis",
            "color": "#764ba2",
            "icon": "🖼️",
        },
        {
            "name": "WGAN",
            "year": 2017,
            "loss": "Wasserstein Distance",
            "pros": "Stable loss, meaningful metric, no mode collapse",
            "cons": "Slow convergence, weight clipping artifacts",
            "use": "Stable training on difficult distributions",
            "color": "#a855f7",
            "icon": "⚖️",
        },
        {
            "name": "WGAN-GP",
            "year": 2017,
            "loss": "Wasserstein + Gradient Penalty",
            "pros": "Very stable, high quality, no weight clipping",
            "cons": "Slower training, higher memory",
            "use": "Production-grade image synthesis",
            "color": "#ec4899",
            "icon": "🎯",
        },
        {
            "name": "StyleGAN",
            "year": 2019,
            "loss": "Non-saturating + R1 penalty",
            "pros": "Photorealistic, style control, disentangled latent",
            "cons": "Extremely compute-heavy, complex architecture",
            "use": "Faces, art, high-res synthesis",
            "color": "#f59e0b",
            "icon": "✨",
        },
        {
            "name": "Conditional GAN",
            "year": 2014,
            "loss": "BCE + class conditioning",
            "pros": "Controllable generation, class-specific output",
            "cons": "Requires labeled data",
            "use": "Class-conditional image generation",
            "color": "#10b981",
            "icon": "🎛️",
        },
    ]

    for i in range(0, len(variants), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(variants):
                v = variants[i + j]
                with col:
                    st.markdown(
                        f"""
                    <div class='card' style='border-top:4px solid {v["color"]};'>
                      <div style='display:flex; align-items:center; gap:.5rem; margin-bottom:.6rem;'>
                        <span style='font-size:1.6rem;'>{v["icon"]}</span>
                        <div>
                          <div style='font-weight:800; color:{v["color"]}; font-size:1rem;'>{v["name"]}</div>
                          <div style='font-size:.75rem; color:#64748b;'>📅 {v["year"]}</div>
                        </div>
                      </div>
                      <div style='font-size:.8rem; margin-bottom:.5rem;'>
                        <span style='color:#64748b;'>Loss:</span>
                        <span style='color:#94a3b8; font-family:monospace;'> {v["loss"]}</span>
                      </div>
                      <div style='background:rgba(16,185,129,.08); border-radius:8px;
                                  padding:.5rem; margin-bottom:.4rem; font-size:.82rem;'>
                        <strong style='color:#34d399;'>✅ Pros:</strong>
                        <span style='color:#94a3b8;'> {v["pros"]}</span>
                      </div>
                      <div style='background:rgba(239,68,68,.08); border-radius:8px;
                                  padding:.5rem; margin-bottom:.4rem; font-size:.82rem;'>
                        <strong style='color:#f87171;'>❌ Cons:</strong>
                        <span style='color:#94a3b8;'> {v["cons"]}</span>
                      </div>
                      <div style='background:rgba(102,126,234,.08); border-radius:8px;
                                  padding:.5rem; font-size:.82rem;'>
                        <strong style='color:#93c5fd;'>🎯 Use Case:</strong>
                        <span style='color:#94a3b8;'> {v["use"]}</span>
                      </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

    # Radar chart comparison
    st.markdown("#### 📡 Variant Capability Radar")
    categories = [
        "Stability",
        "Image Quality",
        "Speed",
        "Simplicity",
        "Controllability",
        "Scalability",
    ]
    scores = {
        "Vanilla GAN": [3, 3, 5, 5, 2, 2],
        "DCGAN": [4, 4, 4, 4, 2, 3],
        "WGAN-GP": [5, 4, 3, 3, 2, 4],
        "StyleGAN": [4, 5, 1, 1, 5, 5],
        "CGAN": [4, 4, 4, 3, 5, 3],
    }
    fig_radar = go.Figure()
    colors_r = ["#667eea", "#a78bfa", "#ec4899", "#f59e0b", "#10b981"]
    for (name, vals), color in zip(scores.items(), colors_r):
        # Convert hex color to rgba with alpha 0.08 for fillcolor
        if color.startswith("#"):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fillcolor = f"rgba({r},{g},{b},0.08)"
        else:
            fillcolor = color

        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                line=dict(color=color, width=2),
                fillcolor=fillcolor,
            )
        )
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 5], gridcolor="rgba(255,255,255,0.1)"
            ),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(13,11,30,0.5)",
        ),
        **make_plotly_layout(420, "GAN Variants — Capability Radar"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 8 — TIPS & TRICKS                                   ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[7]:
    st.markdown(
        "<div class='section-title'>💡 GAN Training Tips & Best Practices</div>",
        unsafe_allow_html=True,
    )

    tips = [
        (
            "⚖️",
            "Balance G and D",
            "#667eea",
            "Never let one network dominate. If D loss → 0 quickly, G receives no gradient "
            "signal. Use a lower LR for D, or update G 2× per D step. Monitor D(real) ≈ D(fake) ≈ 0.5.",
        ),
        (
            "📏",
            "Normalize Your Data",
            "#764ba2",
            "Always normalize inputs to [-1, 1] and use Tanh in Generator output. "
            "Unnormalized data leads to gradient explosions and training failure.",
        ),
        (
            "🏷️",
            "Label Smoothing",
            "#a855f7",
            "Replace real labels (1.0) with soft labels (0.9). Prevents Discriminator from "
            "becoming overconfident too early, improving gradient flow to Generator.",
        ),
        (
            "🔁",
            "Batch Normalization",
            "#ec4899",
            "Apply BatchNorm to all Generator layers and most Discriminator layers. "
            "Exceptions: D input layer and G output layer. Critical for training stability.",
        ),
        (
            "🎲",
            "Add Noise to D Inputs",
            "#f59e0b",
            "Inject small Gaussian noise into Discriminator inputs in early training. "
            "Acts as a regularizer, prevents D from memorizing and overpowering G.",
        ),
        (
            "📊",
            "Track FID Score",
            "#10b981",
            "Fréchet Inception Distance (FID) measures both quality AND diversity of generated "
            "samples. Lower is better. Don't rely only on visual inspection or loss values.",
        ),
        (
            "🔄",
            "Use WGAN-GP",
            "#06b6d4",
            "Switch from vanilla BCE loss to Wasserstein loss + gradient penalty (λ=10). "
            "This eliminates mode collapse and provides a stable, meaningful training signal.",
        ),
        (
            "💾",
            "Save Checkpoints Often",
            "#8b5cf6",
            "GAN training is non-deterministic. Save G and D checkpoints every N epochs. "
            "The best checkpoint is often not the last one — monitor FID and save the best.",
        ),
        (
            "🐌",
            "Use Adam with β₁=0.5",
            "#ec4899",
            "Standard Adam uses β₁=0.9, but GAN training benefits from β₁=0.5 to reduce "
            "momentum and prevent oscillations. Learning rate: 0.0002 is a safe default.",
        ),
        (
            "🔢",
            "Latent Dim Sweet Spot",
            "#a855f7",
            "A latent dimension of 100-256 works well for most datasets. Too small = limited "
            "capacity; too large = sparse space, harder to sample good outputs from.",
        ),
    ]

    for i in range(0, len(tips), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(tips):
                icon, title, color, desc = tips[i + j]
                with col:
                    st.markdown(
                        f"""
                    <div class='tip-card' style='border-left:4px solid {color};
                                                  background:rgba(255,255,255,0.03);'>
                      <div style='display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;'>
                        <span style='font-size:1.4rem;'>{icon}</span>
                        <strong style='color:{color}; font-size:.95rem;'>{title}</strong>
                      </div>
                      <p style='font-size:.85rem; color:#94a3b8; margin:0; line-height:1.65;'>
                        {desc}
                      </p>
                    </div>""",
                        unsafe_allow_html=True,
                    )

    # Quick reference table
    st.markdown("#### 📋 Quick Hyperparameter Reference")
    st.markdown(
        """
    <div class='card'>
    <table style='width:100%; font-size:.88rem; border-collapse:collapse;'>
      <thead>
        <tr style='border-bottom:1px solid rgba(139,92,246,0.3);'>
          <th style='text-align:left; padding:.6rem; color:#a78bfa;'>Parameter</th>
          <th style='text-align:left; padding:.6rem; color:#a78bfa;'>Recommended</th>
          <th style='text-align:left; padding:.6rem; color:#a78bfa;'>Notes</th>
        </tr>
      </thead>
      <tbody>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
          <td style='padding:.5rem; color:#e2e8f0;'>Learning Rate</td>
          <td style='padding:.5rem; color:#34d399; font-family:monospace;'>0.0002</td>
          <td style='padding:.5rem; color:#94a3b8;'>Adam optimizer sweet spot for GANs</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
          <td style='padding:.5rem; color:#e2e8f0;'>Batch Size</td>
          <td style='padding:.5rem; color:#34d399; font-family:monospace;'>64–128</td>
          <td style='padding:.5rem; color:#94a3b8;'>Balance stability vs diversity</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
          <td style='padding:.5rem; color:#e2e8f0;'>Latent Dim</td>
          <td style='padding:.5rem; color:#34d399; font-family:monospace;'>100</td>
          <td style='padding:.5rem; color:#94a3b8;'>Standard starting point</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
          <td style='padding:.5rem; color:#e2e8f0;'>Adam β₁</td>
          <td style='padding:.5rem; color:#34d399; font-family:monospace;'>0.5</td>
          <td style='padding:.5rem; color:#94a3b8;'>Lower than default 0.9</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
          <td style='padding:.5rem; color:#e2e8f0;'>Label Smoothing</td>
          <td style='padding:.5rem; color:#34d399; font-family:monospace;'>0.9 (real)</td>
          <td style='padding:.5rem; color:#94a3b8;'>Soft labels stabilize D</td>
        </tr>
        <tr>
          <td style='padding:.5rem; color:#e2e8f0;'>LeakyReLU slope</td>
          <td style='padding:.5rem; color:#34d399; font-family:monospace;'>0.2</td>
          <td style='padding:.5rem; color:#94a3b8;'>Prevents dead neurons in D</td>
        </tr>
      </tbody>
    </table>
    </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown(
    """
<div class='footer'>
  <div style='font-size:1.15rem; font-weight:800; color:#a78bfa; margin-bottom:.5rem;'>
    🧠 GAN Explorer
  </div>
  <div style='color:#cbd5e1; margin-bottom:.3rem;'>
    Built with ❤️ by <strong>Hafsa Ibrahim</strong> — AI / ML Engineer
  </div>
  <div style='margin: .7rem 0;'>
    <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank'>🔵 LinkedIn</a>
    <a href='https://github.com/HafsaIbrahim5' target='_blank'>⚫ GitHub</a>
  </div>
  <div style='font-size:.75rem; color:#475569; margin-top:.5rem;'>
    GAN Explorer v2.0 · Streamlit · NumPy · Matplotlib · Plotly
    <br>© 2026 Hafsa Ibrahim · Open Source on GitHub
  </div>
</div>
""",
    unsafe_allow_html=True,
)
