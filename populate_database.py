# import argparse
# import os
# import shutil
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from get_embedding_function import get_embedding_function
# from langchain_chroma import Chroma
# #from langchain.vectorstores.chroma import Chroma


# CHROMA_PATH = "chroma"
# DATA_PATH = "data"


# def main():

#     # Check if the database should be cleared (using the --clear flag).
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
#     if args.reset:
#         print("\n✨ Clearing Database\n")
#         clear_database()

#     # Create (or update) the data store.
#     documents = load_documents()
#     chunks = split_documents(documents)
#     add_to_chroma(chunks)


# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()


# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)


# def add_to_chroma(chunks: list[Document]):
#     # Load the existing database.
#     db = Chroma(
#         persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
#     )

#     # Calculate Page IDs.
#     chunks_with_ids = calculate_chunk_ids(chunks)

#     # Add or Update the documents.
#     existing_items = db.get(include=[])  # IDs are always included by default
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}\n\n")

#     # Only add documents that don't exist in the DB.
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if len(new_chunks):
#         print(f"👉 Adding new documents: {len(new_chunks)}\n\n")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#         #db.persist()
#     else:
#         print("✅ No new documents to add\n\n")


# def calculate_chunk_ids(chunks):

#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         # If the page ID is the same as the last one, increment the index.
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         # Add it to the page meta-data.
#         chunk.metadata["id"] = chunk_id

#     return chunks



# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)


# if __name__ == "__main__":
#     main()



# populate_database.py

import os
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the chunk's metadata.
        chunk.metadata["id"] = chunk_id

    return chunks


def process_pdfs_and_populate_database(filepaths):
    # Load documents from the PDF files
    # Ensure CHROMA_PATH exists
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH, exist_ok=True)
    documents = []
    for filepath in filepaths:
        loader = PyPDFLoader(filepath)
        documents.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Calculate chunk IDs
    chunks = calculate_chunk_ids(chunks)

    # Create embeddings and store in Chroma
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Check for existing IDs in the database
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])

    # Only add new chunks that don't exist in the DB
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks to the database.")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new chunks to add.")

    # Optional: Persist the database to disk
    # db.persist()


if __name__ == "__main__":
    # Optional: Add code here if you want to run this script independently.
    pass
