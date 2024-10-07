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
import pdf2image
from datetime import date

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

# os directory
output_directory_path = "converted_images"
if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# User Interface
st.set_page_config(layout="wide")
st.markdown("""## Professional Indemnity Insurance Underwriting System""")
st.write(
    "(For 2024 IFoA GIRO Presentation)"
)

# Functions
def empty_directory(directory_path):
    # List all the files and subdirectories in the given directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        try:
            # Check if it's a file and remove it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            # If it's a directory, remove it using shutil.rmtree
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def get_pdf_to_image(docs):
    if docs is not None:
        images = pdf2image.convert_from_bytes(docs.read())
        for i, image in enumerate(images):
            # Save the image
            image_path = os.path.join(output_directory_path, f'page_{i+1}.png')
            image.save(image_path)
    return images

def main():
    empty_directory(output_directory_path)
    # right hand side UI configuration 
    name_insured = st.text_input("Name under which business is conducted: (‘You’)", key="name_insured")
    address = st.text_input("Addresses of all of your offices & percentage of total fees in each", key="address")
    activity = st.text_input("Give full details of activities you undertake and of any intended change in these", key="activity")
    df = pd.DataFrame(columns=['name', 'age', 'qualifications', 'date_qualified', 'numbers_of_years_in_this_capacity_with_the_proposer'])
    config = {
        'name' : st.column_config.TextColumn('Full Name', required=True),
        'age' : st.column_config.NumberColumn('Age (years)', min_value=0, max_value=122),
        'qualifications' : st.column_config.TextColumn('Qualifications', required=True),
        'date_qualified': st.column_config.DateColumn('Date Qualified', min_value=date(2000, 1, 1), max_value=date(2099, 1, 1), format="DD/MM/YYYY", step=1),
        'numbers_of_years_in_this_capacity_with_the_proposer': st.column_config.NumberColumn('# of Years in this capacity with the proposer', min_value=0, max_value=122, width='large')
    }
    st.markdown("<p style='font-size:12px; color:black;'>Give details below of all Principals (including details of sole principal)</p>", unsafe_allow_html=True)
    #st.markdown("###### Give details below of all Principals (including details of sole principal)")
    st.data_editor(df, column_config = config, num_rows= "dynamic")
    # left hand side activities
    with st.sidebar:
        st.title("Menu:")
        docs = st.file_uploader('Upload your document:', type="pdf")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                get_pdf_to_image(docs)
                #text_chunks = get_text_chunks(raw_text)
                #get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
