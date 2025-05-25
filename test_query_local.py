import os
from utils import search_query
from gpt_service import generate_answer_with_gpt

# Dynamically resolve full paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GLOVE_PATH = os.path.join(DATA_DIR, "Law2Vec.200d.txt")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "ppc_section_embeddings2.json")
SECTION_DATA_PATH = os.path.join(DATA_DIR, "ppc_data_final.json")

def main():
    print("🔎 Working from:", BASE_DIR)
    print("🔎 Checking file paths:")
    print("Exists Law2Vec:", os.path.exists(GLOVE_PATH))
    print("Exists Embeddings:", os.path.exists(EMBEDDINGS_PATH))
    print("Exists Section Data:", os.path.exists(SECTION_DATA_PATH))

    query = input("🔎 Enter your legal query: ")

    top_sections = search_query(query, GLOVE_PATH, EMBEDDINGS_PATH, SECTION_DATA_PATH)

    if top_sections:
        print("\n📘 Top Matching Sections:")
        for sec in top_sections:
            print(f"- Section {sec.get('section', 'N/A')}: {sec.get('offence', '⚠️ Missing')}")

        print("\n🧠 GPT is generating an answer...")
        answer = generate_answer_with_gpt(query, top_sections)
        print("\n✅ GPT Answer:\n", answer)
    else:
        print("⚠️ No relevant sections found. Skipping GPT step.")

if __name__ == "__main__":
    main()
