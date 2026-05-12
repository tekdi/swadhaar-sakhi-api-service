# Swadhaar_AI_ Training _Data.xlsx

import argparse
import json
import re
from typing import (
    Dict,
    List
)
import pandas as pd
import marqo
from langchain.docstore.document import Document
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_finance_keywords(file_path: str) -> List[str]:
    """Extract all finance keywords from English, Hindi, and Hinglish columns in the Excel."""
    df = pd.read_excel(file_path)
    keywords = []
    for col in ('EnglishKeywords', 'HindiKeywords', 'HinglishKeywords'):
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            for kw in str(val).split(','):
                kw = kw.strip()
                if kw:
                    keywords.append(kw)
    return list(dict.fromkeys(keywords))  # deduplicate, preserve order


def store_keyword_index(marqo_client, keywords: List[str], index_name: str, embed_model: str):
    """Delete and recreate a Marqo keyword index, then store all finance keywords."""
    index_settings = {
        "treatUrlsAndPointersAsImages": False,
        "model": embed_model,
        "normalizeEmbeddings": True,
        "textPreprocessing": {
            "splitLength": 1,
            "splitOverlap": 0,
            "splitMethod": "passage"
        }
    }
    try:
        marqo_client.index(index_name).delete()
        print(f"Existing keyword index '{index_name}' deleted.")
    except Exception:
        pass
    marqo_client.create_index(index_name, settings_dict=index_settings)
    print(f"Keyword index '{index_name}' created.")

    batch_size = 50
    docs = [{"text": kw} for kw in keywords]
    for chunk in [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]:
        marqo_client.index(index_name).add_documents(
            documents=chunk, client_batch_size=batch_size, tensor_fields=["text"])
    print(f"Stored {len(keywords)} keywords in index '{index_name}'.")

def load_documents(file_path, input_chunk_size, input_chunk_overlap, search_type='dense'):

    docs: List[Dict[str, str]] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=input_chunk_size, chunk_overlap=input_chunk_overlap)
    pattern = re.compile(r'[\x00-\x1F\x7F\u00A0]')

    df = pd.read_excel(file_path)

    videos = []

    for i, row in df.iterrows():
        topic = row['Topic']
        VideoUrl = row['VideoUrl']
        VideoDesc= row['VideoDesc']
        videos.append(VideoUrl)
        metadata={
                "page_label": "",
                "file_name": "",
                "file_path": "",
                "file_type": ""
            }
        for chunk in splitter.split_text(re.sub(pattern, '', re.sub(r'\s*\n\s*', ' ', row['Content']))):
            text = f"{topic}: \n {chunk} \n Video Link of {topic} ({VideoDesc}): {VideoUrl}"
            doc = {
                "text": text,
                "metadata": json.dumps(metadata) if metadata else json.dumps({})
            }
            if search_type == 'hybrid':
                # Collect keywords from all three language columns.
                # These are NOT added to tensor_fields so they only contribute
                # to the lexical (BM25) leg of hybrid search, keeping dense
                # embeddings clean from keyword noise.
                keyword_parts = []
                for col in ('EnglishKeywords', 'HindiKeywords', 'HinglishKeywords'):
                    val = row.get(col, '')
                    if pd.notna(val) and str(val).strip():
                        keyword_parts.append(str(val).strip())
                if keyword_parts:
                    doc['keywords'] = ', '.join(keyword_parts)
            docs.append(doc)
    
    f1 = open("indexed_videos.txt", "w")
    f1.write(str(videos))
    f1.close()
    return docs

def chunk_list(document, batch_size):
    """Return a list of batch sized chunks from document."""
    return [document[i: i + batch_size] for i in range(0, len(document), batch_size)]


def indexer_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marqo_url',
                        type=str,
                        required=True,
                        help='Endpoint URL of marqo',
                        )
    parser.add_argument('--index_name',
                        type=str,
                        required=True,
                        help='Name of marqo index',
                        )
    parser.add_argument('--folder_path',
                        type=str,
                        required=True,
                        help='Path to the folder',
                        default="input_data"
                        )
    parser.add_argument('--embedding_model',
                        type=str,
                        required=False,
                        help='data embedding model to be used.',
                        default="flax-sentence-embeddings/all_datasets_v4_mpnet-base"
                        )
    parser.add_argument('--chunk_size',
                        type=int,
                        required=False,
                        help='documents chunk size',
                        default=1024
                        )
    parser.add_argument('--chunk_overlap',
                        type=int,
                        required=False,
                        help='documents chunk size',
                        default=200
                        )
    parser.add_argument('--split_length',
                        type=int,
                        required=False,
                        help='pre-processing split_length',
                        default=1
                        )
    parser.add_argument('--split_overlap',
                        type=int,
                        required=False,
                        help='pre-processing split_overlap',
                        default=0
                        )
    parser.add_argument('--fresh_index',
                        action='store_true',
                        help='Is the indexing fresh'
                        )
    parser.add_argument('--search_type',
                        type=str,
                        required=False,
                        choices=['dense', 'hybrid'],
                        default='dense',
                        help='Type of search the index will support: dense (tensor only) or hybrid (tensor + lexical/BM25). '
                             'Marqo 2.x unstructured indexes support both; this flag documents intent and adjusts index settings.'
                        )
    parser.add_argument('--keyword_index_name',
                        type=str,
                        required=False,
                        default=None,
                        help='Name of a separate Marqo index for finance keywords (used as pre-LLM intent gate). '
                             'Always recreated fresh. Omit to skip keyword indexing.'
                        )

    args = parser.parse_args()

    MARQO_URL = args.marqo_url
    MARQO_INDEX_NAME = args.index_name
    FOLDER_PATH = args.folder_path
    FRESH_INDEX = args.fresh_index
    EMBED_MODEL = args.embedding_model
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    SPLIT_LENGTH = args.split_length
    SPLIT_OVERLAP = args.split_overlap
    SEARCH_TYPE = args.search_type
    KEYWORD_INDEX = args.keyword_index_name

    if SEARCH_TYPE == 'hybrid':
        # Marqo 2.x requires a STRUCTURED index for hybrid search.
        # allFields must declare every field the documents will contain.
        # tensorFields drives dense (HNSW) embeddings; lexical_search feature
        # enables BM25 on that field. keywords is lexical-only (no tensor).
        index_settings = {
            "type": "structured",
            "model": EMBED_MODEL,
            "normalizeEmbeddings": True,
            "allFields": [
                {"name": "text",     "type": "text", "features": ["lexical_search"]},
                {"name": "keywords", "type": "text", "features": ["lexical_search"]},
                {"name": "metadata", "type": "text"},
            ],
            "tensorFields": ["text"],
        }
    else:
        # Dense: unstructured index, unchanged from original
        index_settings = {
            "treatUrlsAndPointersAsImages": False,
            "model": EMBED_MODEL,
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": SPLIT_LENGTH,
                "splitOverlap": SPLIT_OVERLAP,
                "splitMethod": "passage"
            }
        }

    print(f"Search type: {SEARCH_TYPE.upper()}")
    if SEARCH_TYPE == 'hybrid':
        print("Hybrid mode: structured index with tensor (text) + lexical BM25 (text, keywords).")

    # Initialize Marqo instance
    marqo_client = marqo.Client(url=MARQO_URL)
    if FRESH_INDEX:
        try:
            marqo_client.index(MARQO_INDEX_NAME).delete()
            print("Existing Index successfully deleted.")
        except:
            print("Index does not exist. Creating new index")

        marqo_client.create_index(
            MARQO_INDEX_NAME, settings_dict=index_settings)
        print(f"Index {MARQO_INDEX_NAME} created ({SEARCH_TYPE} search type).")

    print("Loading documents...")
    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_TYPE)

    f = open("indexed_documents.txt", "w")
    f.write(str(documents))
    f.close()

    print(f"Indexing {len(documents)} document chunks...")

    _document_batch_size = 50
    chunks = list(chunk_list(documents, _document_batch_size))

    for chunk in chunks:
        if SEARCH_TYPE == 'hybrid':
            # The Python client 2.1.0 always includes tensorFields or nonTensorFields in the
            # request body, but Marqo 2.x structured indexes reject both as extra fields.
            # Direct HTTP bypasses the client and sends only the documents, which is what
            # structured indexes expect (tensor fields are already defined in the schema).
            url = f"{MARQO_URL}/indexes/{MARQO_INDEX_NAME}/documents"
            response = requests.post(url, json={"documents": chunk}, timeout=120)
            if not response.ok:
                print(f"Batch indexing error {response.status_code}: {response.text}")
            response.raise_for_status()
        else:
            marqo_client.index(MARQO_INDEX_NAME).add_documents(
                documents=chunk, client_batch_size=_document_batch_size, tensor_fields=['text'])

    print(f"============ INDEX DONE ({SEARCH_TYPE.upper()}) =============")

    if KEYWORD_INDEX:
        print("\nExtracting finance keywords from Excel...")
        keywords = extract_finance_keywords(FOLDER_PATH)
        print(f"Extracted {len(keywords)} unique keywords.")
        store_keyword_index(marqo_client, keywords, KEYWORD_INDEX, EMBED_MODEL)
        print("============ KEYWORD INDEX DONE =============")


if __name__ == "__main__":
    indexer_main()
    
# RUN (dense - default)
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx --fresh_index

# RUN (hybrid)
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx --search_type=hybrid
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx --search_type=hybrid --fresh_index

# dev (dense)
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar_dev --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar_dev --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx --fresh_index

# dev (hybrid)
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar_dev --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx --search_type=hybrid
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar_dev --folder_path=/path/to/Swadhaar_AI_Training_Data.xlsx --search_type=hybrid --fresh_index