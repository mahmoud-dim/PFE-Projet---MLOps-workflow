# import json
# import os
# import faiss
# import numpy as np
# import pickle
# from langchain_ollama import OllamaEmbeddings

# # ============================================================
# # build_faiss_index_hn.py
# # Script à exécuter UNE SEULE FOIS pour indexer la KB FAISS
# # Embeddings : Ollama local (nomic-embed-text)
# # ============================================================

# embeddings_model = OllamaEmbeddings(
#     model="nomic-embed-text",
#     base_url="http://localhost:11434"
# )

# # ----------------------------------------
# # Étape 1 : Charger le JSON
# # ----------------------------------------
# BASE_DIR  = os.path.dirname(__file__)
# json_path = os.path.join(BASE_DIR, "../data/knowledge_base_huawei_nokia.json")

# print("📂 Chargement de la Knowledge Base Huawei/Nokia...")
# with open(json_path, "r", encoding="utf-8") as f:
#     causes_data = json.load(f)

# print(f"✅ {len(causes_data)} causes chargées depuis la KB.")

# # ----------------------------------------
# # Préparer les documents pour embeddings
# # ----------------------------------------
# documents = []
# for item in causes_data:
#     text = (
#         f"Cause: {item['Cause']}\n"
#         f"Indicateurs: {', '.join(item['Indicators'])}\n"
#         f"Description: {item['Description']}"
#     )
#     documents.append(text)

# print(f"\n📄 {len(documents)} documents préparés :")
# for i, doc in enumerate(documents):
#     print(f"  [{i}] {causes_data[i]['Cause']}")

# # ----------------------------------------
# # Étape 2 : Générer les embeddings avec Ollama
# # ----------------------------------------
# print("\n⏳ Génération des embeddings avec Ollama (nomic-embed-text)...")

# embeddings = embeddings_model.embed_documents(documents)

# print(f"✅ {len(embeddings)} vecteurs de dimension {len(embeddings[0])}")

# # ----------------------------------------
# # Étape 3 : Créer l'index FAISS
# # ----------------------------------------
# print("\n🔧 Création de l'index FAISS...")

# dimension = len(embeddings[0])
# index     = faiss.IndexFlatL2(dimension)
# emb_array = np.array(embeddings).astype("float32")
# index.add(emb_array)

# print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs.")

# # ----------------------------------------
# # Étape 4 : Sauvegarder
# # ----------------------------------------
# os.makedirs(os.path.join(BASE_DIR, "../data"), exist_ok=True)

# faiss_path = os.path.join(BASE_DIR, "../data/faiss_index_huawei_nokia.index")
# faiss.write_index(index, faiss_path)
# print(f"\n💾 Index FAISS : {faiss_path}")

# chunks_path = os.path.join(BASE_DIR, "../data/chunks_huawei_nokia.pkl")
# with open(chunks_path, "wb") as f:
#     pickle.dump(documents, f)
# print(f"💾 Chunks      : {chunks_path}")

# print("\n" + "="*55)
# print("✅ INDEXATION TERMINÉE")
# print("="*55)
# print(f"  KB         : {len(causes_data)} causes indexées")
# print(f"  Modèle     : Ollama nomic-embed-text (local)")
# print(f"  Dimension  : {dimension}")
# print("\n💡 Prochaine étape : exécuter main_hn.py")


import json
import os
import faiss
import numpy as np
import pickle
import boto3
from io import BytesIO
from langchain_ollama import OllamaEmbeddings

# ============================================================
# build_faiss_index_hn.py
# Embeddings : Ollama local (nomic-embed-text)
# Input  : MinIO → knowledge-base/knowledge_base_huawei_nokia.json
# Output : MinIO → models/scenario2/faiss_index_huawei_nokia.index
#                  models/scenario2/chunks_huawei_nokia.pkl
# ============================================================

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://10.98.20.211:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_KB     = "knowledge-base"
BUCKET_MODELS = "models"

KB_KEY     = "knowledge_base_huawei_nokia.json"
FAISS_KEY  = "scenario2/faiss_index_huawei_nokia.index"
CHUNKS_KEY = "scenario2/chunks_huawei_nokia.pkl"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

s3 = boto3.client(
    "s3",
    endpoint_url          = MINIO_ENDPOINT,
    aws_access_key_id     = MINIO_ACCESS_KEY,
    aws_secret_access_key = MINIO_SECRET_KEY
)

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

# ----------------------------------------
# Étape 1 : Charger la KB depuis MinIO
# ----------------------------------------
print("📂 Chargement de la Knowledge Base depuis MinIO...")
response    = s3.get_object(Bucket=BUCKET_KB, Key=KB_KEY)
causes_data = json.loads(response["Body"].read().decode("utf-8"))
print(f"✅ {len(causes_data)} causes chargées.")

# ----------------------------------------
# Préparer les documents
# ----------------------------------------
documents = []
for item in causes_data:
    text = (
        f"Cause: {item['Cause']}\n"
        f"Indicateurs: {', '.join(item['Indicators'])}\n"
        f"Description: {item['Description']}"
    )
    documents.append(text)

print(f"\n📄 {len(documents)} documents préparés :")
for i, doc in enumerate(documents):
    print(f"  [{i}] {causes_data[i]['Cause']}")

# ----------------------------------------
# Étape 2 : Générer les embeddings
# ----------------------------------------
print("\n⏳ Génération des embeddings avec Ollama (nomic-embed-text)...")
embeddings = embeddings_model.embed_documents(documents)
print(f"✅ {len(embeddings)} vecteurs de dimension {len(embeddings[0])}")

# ----------------------------------------
# Étape 3 : Créer l'index FAISS
# ----------------------------------------
print("\n🔧 Création de l'index FAISS...")
dimension = len(embeddings[0])
index     = faiss.IndexFlatL2(dimension)
emb_array = np.array(embeddings).astype("float32")
index.add(emb_array)
print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs.")

# ----------------------------------------
# Étape 4 : Sauvegarder dans MinIO
# ----------------------------------------
print("\n💾 Sauvegarde dans MinIO...")

# Sauvegarder FAISS index
faiss_buffer = BytesIO()
faiss.write_index(index, "/tmp/faiss_hn.index")
with open("/tmp/faiss_hn.index", "rb") as f:
    s3.put_object(Bucket=BUCKET_MODELS, Key=FAISS_KEY, Body=f.read())
print(f"   ✅ Index FAISS → {FAISS_KEY}")

# Sauvegarder chunks
chunks_buffer = BytesIO()
pickle.dump(documents, chunks_buffer)
chunks_buffer.seek(0)
s3.put_object(Bucket=BUCKET_MODELS, Key=CHUNKS_KEY, Body=chunks_buffer.getvalue())
print(f"   ✅ Chunks      → {CHUNKS_KEY}")

print("\n" + "="*55)
print("✅ INDEXATION TERMINÉE")
print("="*55)
print(f"  KB        : {len(causes_data)} causes indexées")
print(f"  Modèle    : Ollama nomic-embed-text")
print(f"  Dimension : {dimension}")
print(f"  FAISS     : {FAISS_KEY}")
print(f"  Chunks    : {CHUNKS_KEY}")