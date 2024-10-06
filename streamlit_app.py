# Import packages
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
import json
import pandas as pd
from torchvision import transforms
from transformers import AutoModelForObjectDetection
import torch
import openai
import os
import pymupdf

import qdrant_client
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.indices.multi_modal.retriever import MultiModalVectorIndexRetriever

device = "cuda" if torch.cuda.is_available() else "cpu"
openai.api_key = st.secrets["OPENAI_API_KEY"]
output_directory_path = "converted_images"
if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# User Interface
st.markdown("""## Professional Indemnity Insurance Underwriting System""")
st.write(
    "(For 2024 IFoA GIRO Presentation)"
)

def get_pdf_to_image(docs):
    if docs is not None:
        with pymupdf.open(stream=docs.read(), filetype="pdf") as pdf_file:
            pdf_page_count = pdf_file.page_count   
            for page_number in range(pdf_page_count):  
                # Get the page
                page = pdf_file[page_number]
                # Convert the page to an image
                pix = page.get_pixmap()
                # Create a Pillow Image object from the pixmap
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # Save the image
                image_path = os.path.join(output_dir, f'page_{i+1}.png')
                image.save(image_path)
    return documents

def main():
    with st.sidebar:
        st.title("Menu:")
        docs = st.file_uploader('Upload your document:', type="pdf")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_pdf_image = get_pdf_to_image(docs)
                #text_chunks = get_text_chunks(raw_text)
                #get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
