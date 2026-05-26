from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import numpy as np
import pickle
import io

app = FastAPI(title="Karol IA", description="API de identificación de hongos — MycoCOL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cargar modelos al iniciar ─────────────────────────────────────────────────
print("Cargando Karol IA...")

with open("especie2idx.pkl", "rb") as f:
    especie2idx = pickle.load(f)
idx2especie = {v: k for k, v in especie2idx.items()}
num_clases = len(especie2idx)

prototipos   = np.load("prototipos_fewshot.npy")
esp_fewshot  = np.load("especies_fewshot.npy", allow_pickle=True).tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = models.efficientnet_v2_s(weights=None)
modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, num_clases)
modelo.load_state_dict(torch.load("karol_ia_best.pth", map_location=device))
modelo = modelo.to(device)
modelo.eval()

extractor = nn.Sequential(*list(modelo.children())[:-1]).to(device)
extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"✅ Karol IA lista — {num_clases} especies — {device}")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "nombre": "Karol IA",
        "version": "1.0",
        "especies": num_clases,
        "descripcion": "API de identificación micológica — MycoCOL"
    }

@app.post("/identificar")
async def identificar(imagen: UploadFile = File(...)):
    # Leer imagen
    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Predicción modelo principal
        outputs = modelo(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        top5_idx = probs.topk(5).indices.tolist()
        top5_prob = probs.topk(5).values.tolist()

        # Embedding para few-shot
        emb = extractor(tensor).squeeze().cpu().numpy()

    # Predicción principal
    pred_principal = {
        "especie": idx2especie[top5_idx[0]],
        "confianza": round(top5_prob[0], 4),
        "top5": [
            {"especie": idx2especie[i], "confianza": round(p, 4)}
            for i, p in zip(top5_idx, top5_prob)
        ]
    }

    # Predicción few-shot
    sims = cosine_similarity(emb.reshape(1, -1), prototipos)[0]
    top3_fs = np.argsort(sims)[::-1][:3]
    pred_fewshot = [
        {"especie": esp_fewshot[i], "similitud": round(float(sims[i]), 4)}
        for i in top3_fs
    ]

    return {
        "modelo_principal": pred_principal,
        "few_shot": pred_fewshot,
        "dispositivo": str(device)
    }

@app.get("/especies")
def listar_especies():
    return {"total": num_clases, "especies": list(especie2idx.keys())}