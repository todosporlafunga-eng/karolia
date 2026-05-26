import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import io
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Karol IA — MycoCOL",
    page_icon="te amito karol🍄",
    layout="centered"
)

# ── Cargar modelo desde Hugging Face ─────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    REPO = "todosporlafunga/karol-ia"
    
    with st.spinner("Cargando Karol IA..."):
        pth          = hf_hub_download(REPO, "karol_ia_best.pth")
        pkl_idx      = hf_hub_download(REPO, "especie2idx.pkl")
        proto        = hf_hub_download(REPO, "prototipos_fewshot.npy")
        esp_fs       = hf_hub_download(REPO, "especies_fewshot.npy")

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

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🍄 Karol IA")
st.subheader("Identificación de hongos — MycoCOL")
st.markdown("Sube una foto de un hongo y Karol IA lo identificará.")

opcion = st.radio(
    "¿Cómo quieres ingresar la imagen?",
    ["📁 Subir desde galería", "📷 Tomar foto"],
    horizontal=True
)

archivo = None

if opcion == "📁 Subir desde galería":
    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
else:
    archivo = st.camera_input("Tomar foto")

if archivo:
    img = Image.open(archivo).convert("RGB")
    st.image(img, caption="Imagen cargada", use_container_width=True)

    with st.spinner("Identificando..."):
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = modelo(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            top5_idx  = probs.topk(5).indices.tolist()
            top5_prob = probs.topk(5).values.tolist()
            emb = extractor(tensor).squeeze().cpu().numpy()

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

    sims = cosine_similarity(emb.reshape(1,-1), prototipos)[0]
    top3 = np.argsort(sims)[::-1][:3]
    st.markdown("#### Especies similares (few-shot)")
    for i in top3:
        nombre = esp_fewshot[i].replace("_", " ").title()
        sim    = round(float(sims[i]) * 100, 1)
        st.markdown(f"- **{nombre}** — similitud {sim}%")

st.markdown("---")
st.caption("MycoCOL — Plataforma de micología colombiana | Karol IA v1.0")