import json
import os
import faiss
import numpy as np
import pickle
import cohere
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
# load_dotenv()


# ============================================================
# build_faiss_index_hn.py
# Script à exécuter UNE SEULE FOIS pour indexer la KB FAISS
# Embeddings : Cohere API (cloud, gratuit)
# ============================================================

co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ----------------------------------------
# Étape 1 : Charger le JSON
# ----------------------------------------
json_path = os.path.join(os.path.dirname(__file__), "../data/knowledge_base_huawei_nokia.json")

print("📂 Chargement de la Knowledge Base...")
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
# Étape 2 : Générer les embeddings avec Cohere
# input_type="search_document" pour l'indexation
# ----------------------------------------
print("\n⏳ Génération des embeddings avec Cohere API...")

response = co.embed(
    texts=documents,
    model="embed-multilingual-v3.0",    # multilingue : français ✅
    input_type="search_document"         # obligatoire pour indexation
)

embeddings = response.embeddings
print(f"✅ {len(embeddings)} vecteurs de dimension {len(embeddings[0])}")

# ----------------------------------------
# Étape 3 : Créer l'index FAISS
# ----------------------------------------
print("\n🔧 Création de l'index FAISS...")

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
emb_array = np.array(embeddings).astype("float32")
index.add(emb_array)

print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs.")

# ----------------------------------------
# Étape 4 : Sauvegarder
# ----------------------------------------
os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)

faiss_path = os.path.join(os.path.dirname(__file__), "../data/faiss_index_huawei_nokia.index")
faiss.write_index(index, faiss_path)
print(f"\n💾 Index FAISS : {faiss_path}")

chunks_path = os.path.join(os.path.dirname(__file__), "../data/chunks_huawei_nokia.pkl")
with open(chunks_path, "wb") as f:
    pickle.dump(documents, f)
print(f"💾 Chunks      : {chunks_path}")

print("\n" + "="*55)
print("✅ INDEXATION TERMINÉE")
print("="*55)
print(f"  KB         : {len(causes_data)} causes indexées")
print(f"  Modèle     : Cohere embed-multilingual-v3.0")
print(f"  Dimension  : {dimension}")
print("\n💡 Prochaine étape : exécuter main_hn.py")