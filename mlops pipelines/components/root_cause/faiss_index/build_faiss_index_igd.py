import json
import os
import faiss
import numpy as np
import pickle
from langchain_ollama import OllamaEmbeddings

# ============================================================
# build_faiss_index_igd.py
# Script à exécuter UNE SEULE FOIS pour indexer la KB FAISS IGD
# Embeddings : Ollama local (nomic-embed-text)
# ============================================================

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# ----------------------------------------
# Étape 1 : Charger le JSON
# ----------------------------------------
BASE_DIR  = os.path.dirname(__file__)
json_path = os.path.join(BASE_DIR, "../data/knowledge_base_igd.json")

print("📂 Chargement de la Knowledge Base IGD...")
with open(json_path, "r", encoding="utf-8") as f:
    causes_data = json.load(f)

print(f"✅ {len(causes_data)} causes chargées depuis la KB.")

# ----------------------------------------
# Préparer les documents pour embeddings
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
# Étape 2 : Générer les embeddings avec Ollama
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
# Étape 4 : Sauvegarder
# ----------------------------------------
os.makedirs(os.path.join(BASE_DIR, "../data"), exist_ok=True)

faiss_path = os.path.join(BASE_DIR, "../data/faiss_index_igd.index")
faiss.write_index(index, faiss_path)
print(f"\n💾 Index FAISS : {faiss_path}")

chunks_path = os.path.join(BASE_DIR, "../data/chunks_igd.pkl")
with open(chunks_path, "wb") as f:
    pickle.dump(documents, f)
print(f"💾 Chunks      : {chunks_path}")

print("\n" + "="*55)
print("✅ INDEXATION IGD TERMINÉE")
print("="*55)
print(f"  KB         : {len(causes_data)} causes indexées")
print(f"  Modèle     : Ollama nomic-embed-text (local)")
print(f"  Dimension  : {dimension}")
print("\n💡 Prochaine étape : exécuter main_igd.py")