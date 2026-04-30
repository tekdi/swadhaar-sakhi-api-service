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
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(file_path, input_chunk_size, input_chunk_overlap):

    docs: List[Dict[str, str]] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=input_chunk_size, chunk_overlap=input_chunk_overlap)
    pattern = re.compile(r'[\x00-\x1F\x7F\u00A0]')

    df = pd.read_excel(file_path)

    for i, row in df.iterrows():
        topic = row['Topic']
        VideoUrl = row['VideoUrl']
        VideoDesc= row['VideoDesc']
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
            docs.append(doc)
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
        print(f"Index {MARQO_INDEX_NAME} created.")

    print("Loading documents...")
    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    f = open("indexed_documents.txt", "w")
    f.write(str(documents))
    f.close()

    print(f"Indexing documents...")

    tensor_fields = ['text']
    _document_batch_size = 50
    chunks = list(chunk_list(documents, _document_batch_size))

    for chunk in chunks:
        marqo_client.index(MARQO_INDEX_NAME).add_documents(
            documents=chunk, client_batch_size=_document_batch_size, tensor_fields=tensor_fields)

    print("============ INDEX DONE =============")


if __name__ == "__main__":
    indexer_main()
    
# RUN

# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar --folder_path=/home/ttpl-rt-228/Downloads/Swadhaar_AI_Training_Data.xlsx
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar --folder_path=/home/ttpl-rt-228/Documents/Swadhaar_AI_Training_Data.xlsx --fresh_index (FOR FRESH INDEXING)


#dev
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar_dev --folder_path=/home/ttpl-rt-228/Downloads/Swadhaar_AI_Training_Data.xlsx
# python3 indexer_swadhaar.py --marqo_url=http://0.0.0.0:8882 --index_name=swadhaar_dev --folder_path=/home/ttpl-rt-228/Documents/Swadhaar_AI_Training_Data.xlsx --fresh_index (FOR FRESH INDEXING)