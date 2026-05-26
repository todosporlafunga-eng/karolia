import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import io
import base64
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Karol IA — MycoCOL",
    page_icon="🍄",
    layout="centered"
)

@st.cache_resource
def cargar_modelo():
    REPO = "todosporlafunga/karol-ia"
    with st.spinner("Cargando Karol IA..."):
        pth     = hf_hub_download(REPO, "karol_ia_best.pth")
        pkl_idx = hf_hub_download(REPO, "especie2idx.pkl")
        proto   = hf_hub_download(REPO, "prototipos_fewshot.npy")
        esp_fs  = hf_hub_download(REPO, "especies_fewshot.npy")

    with open(pkl_idx, "rb") as f:
        especie2idx = pickle.load(f)
    idx2especie = {v: k for k, v in especie2idx.items()}
    num_clases  = len(especie2idx)

    prototipos  = np.load(proto)
    esp_fewshot = np.load(esp_fs, allow_pickle=True).tolist()

    device = torch.device("cpu")
    modelo = models.efficientnet_v2_s(weights=None)
    modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, num_clases)
    modelo.load_state_dict(torch.load(pth, map_location=device))
    modelo.eval()

    extractor = nn.Sequential(*list(modelo.children())[:-1])
    extractor.eval()

    return modelo, extractor, idx2especie, especie2idx, prototipos, esp_fewshot, device

modelo, extractor, idx2especie, especie2idx, prototipos, esp_fewshot, device = cargar_modelo()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🍄 Karol IA")
st.subheader("Identificación de hongos — MycoCOL")
st.markdown("Sube una foto de un hongo y Karol IA lo identificará.")

# ── Input nativo HTML para móvil y computadora ────────────────────────────────
st.markdown("""
    <input type="file" id="file-input" accept="image/*" 
        style="display:none">
    <label for="file-input" style="
        display: inline-block;
        padding: 0.6rem 1.2rem;
        background: #ff4b4b;
        color: white;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        margin-bottom: 1rem;">
        📁 Seleccionar imagen
    </label>
    <div id="preview-container" style="margin-top:10px"></div>
    <script>
        const input = document.getElementById('file-input');
        input.addEventListener('change', function() {
            const file = this.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('preview-container').innerHTML = 
                    '<img src="' + e.target.result + '" style="max-width:100%;border-radius:8px;margin-top:8px">';
            };
            reader.readAsDataURL(file);
        });
    </script>
""", unsafe_allow_html=True)

archivo = st.file_uploader(
    "O arrastra una imagen aquí",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible"
)

# ── También opción cámara ─────────────────────────────────────────────────────
usar_camara = st.checkbox("📷 Usar cámara")
if usar_camara:
    archivo = st.camera_input("Tomar foto")

if archivo:
    img = Image.open(archivo).convert("RGB")
    st.image(img, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Identificando..."):
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs   = modelo(tensor)
            probs     = torch.softmax(outputs, dim=1)[0]
            top5_idx  = probs.topk(5).indices.tolist()
            top5_prob = probs.topk(5).values.tolist()
            emb       = extractor(tensor).squeeze().cpu().numpy()

    especie_pred = idx2especie[top5_idx[0]].replace("_", " ").title()
    confianza    = round(top5_prob[0] * 100, 1)

    st.markdown("---")
    st.markdown(f"### 🍄 {especie_pred}")
    st.progress(top5_prob[0])
    st.markdown(f"**Confianza: {confianza}%**")

    st.markdown("#### Top 5 predicciones")
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob)):
        nombre = idx2especie[idx].replace("_", " ").title()
        st.markdown(f"{i+1}. **{nombre}** — {round(prob*100,1)}%")

    sims = cosine_similarity(emb.reshape(1, -1), prototipos)[0]
    top3 = np.argsort(sims)[::-1][:3]
    st.markdown("#### Especies similares (few-shot)")
    for i in top3:
        nombre = esp_fewshot[i].replace("_", " ").title()
        sim    = round(float(sims[i]) * 100, 1)
        st.markdown(f"- **{nombre}** — similitud {sim}%")

st.markdown("---")
st.caption("MycoCOL — Plataforma de micología colombiana | Karol IA v1.0")