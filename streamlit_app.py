# Import packages
import streamlit as st
from streamlit import session_state as ss, rerun as rr

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
from PIL import Image
import fitz
import copy

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
model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
).to(device)
structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(device)

# os directory
output_directory_path = "converted_images"
cropped_table_directory_path = "cropped_tables"
if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)
if not os.path.exists(cropped_table_directory_path):
    os.makedirs(cropped_table_directory_path)


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
        pdf_document = fitz.open(stream=docs.read(), filetype="pdf")

        # Iterate through each page and convert to an image
        for page_number in range(pdf_document.page_count):
            # Get the page
            page = pdf_document[page_number]
            # Convert the page to an image
            pix = page.get_pixmap()
            # Create a Pillow Image object from the pixmap
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Save the image
            image_path = os.path.join(output_directory_path, f'page_{page_number+1}.png')
            image.save(image_path)
    return image

def retrieve_relevant_images(directory_path, query):
    images = SimpleDirectoryReader(directory_path).load_data()
    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path="qdrant_index")
    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")
    storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

    # Create the MultiModal index
    index = MultiModalVectorStoreIndex.from_documents(
        images,
        storage_context=storage_context,
    )

    # Retrieve top 2 images
    retriever_engine = index.as_retriever(image_similarity_top_k=2)
    # retrieve for the query using text to image retrieval
    retrieval_results = retriever_engine.text_to_image_retrieve(query)
    retrieved_images = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_images.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
    return retrieved_images

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32
    )
    return boxes

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
    ]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects

def detect_and_crop_save_table(file_path):
    image = Image.open(file_path)
    detection_transform = transforms.Compose(
        [
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    # postprocess to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)
    for idx in range(len(detected_tables)):
        # crop detected table out of image
        cropped_table = image.crop(detected_tables[idx]["bbox"])
        cropped_table.save(os.path.join(cropped_table_directory_path, f'cropped_table_{idx}.png'))
    return detected_tables

def information_extractor(prompt, image_directory_path, single_image = False):
    if single_image:
        retrieved_images = []
        retrieved_images.append(image_directory_path)
        image_documents = [ImageDocument(image_path=image_path) for image_path in retrieved_images]
    else:
        image_documents = SimpleDirectoryReader(image_directory_path).load_data()
    openai_mm_llm = OpenAIMultiModal(model="gpt-4o-mini", api_key=openai.api_key, max_new_tokens=1500)
    response = openai_mm_llm.complete(
        prompt=prompt,
        image_documents=image_documents,
    )
    return response
    
def main():
    # Empty directories
    empty_directory(output_directory_path)
    empty_directory(cropped_table_directory_path)
    
    # Prompts to feed in
    prompt_1="What's the text inside the third box? "
    prompt_2="on the first image in this collection, simply return me the address with the postal code without the website. Do not return other words not in the document"
    prompt_3="what's the date commenced? just return the date"
    prompt_4="return the information in the table: name, age, qualifications, Date qualified, Numbers of years in this capacity with the Proposer, only return the table in dictionary format (without the python in the response) with key='data', give me consistent response no matter when i ask"
    
    # Right hand side UI configuration 
    df = pd.DataFrame(columns=['name', 'age', 'qualifications', 'date_qualified', 'numbers_of_years_in_this_capacity_with_the_proposer'])
    config = {
        'name' : st.column_config.TextColumn('Full Name', required=True),
        'age' : st.column_config.NumberColumn('Age (years)', min_value=0, max_value=122),
        'qualifications' : st.column_config.TextColumn('Qualifications', required=True),
        'date_qualified': st.column_config.DateColumn('Date Qualified', min_value=date(2000, 1, 1), max_value=date(2099, 1, 1), format="DD/MM/YYYY", step=1),
        'numbers_of_years_in_this_capacity_with_the_proposer': st.column_config.NumberColumn('# of Years in this capacity with the proposer', min_value=0, max_value=122, width='large')
    }
    updated_df = copy.deepcopy(df)
    
    # Left hand side activities
    with st.sidebar:
        st.title("Menu:")
        docs = st.file_uploader('Upload your document:', type="pdf")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                images = get_pdf_to_image(docs)
                # query = "bottom centre footer with '2', contain two tables, with 'Give details below of previous business experience, as appropriate, or attach curricula vitae', with 'Give details below of all Principals (including details of sole principal)', table with column names: name, age, qualifications, Date qualified, Numbers of years in this capacity with the Proposer"
                # retrieved_relevant_images = retrieve_relevant_images(output_directory_path)   ## smarter and dynamically finding the page based on query, but more memory consuming
                # for file_path in retrieved_relevant_images:
                detect_and_crop_save_table(os.path.join(output_directory_path, f'page_2.png'))
                st.info("Extracting information...⌛️")
                # Q1
                response_1 = information_extractor(prompt_1, os.path.join(output_directory_path, f'page_1.png'),single_image = True)
                response_text_1 = response_1.text
                # Q2
                response_2 = information_extractor(prompt_2, os.path.join(output_directory_path))
                response_text_2 = response_2.text
                # Q3
                response_3 = information_extractor(prompt_3, os.path.join(output_directory_path, f'page_1.png'), single_image = True)
                response_text_3 = response_3.text
                # Q4
                response_4 = information_extractor(prompt_4, os.path.join(cropped_table_directory_path))
                response_text_4 = response_4.text                
                updated_df = pd.DataFrame(json.loads(response_text_4)['data'])
                updated_df['date_qualified'] = pd.to_datetime(updated_df['date_qualified'], format='%Y-%m-%d')
                st.success("Done")

    # Right hand side update
    if df.empty and updated_df.empty:
        name_insured = st.text_input("Name under which business is conducted: (‘You’)", key="name_insured")
        address = st.text_input("Addresses of all of your offices & percentage of total fees in each", key="address")
        date = st.date_input("Date Commenced", key="date")
        st.markdown("<p style='font-size:14px; color:black;'>Give details below of all Principals (including details of sole principal)</p>", unsafe_allow_html=True)
        st.data_editor(df, column_config = config,  num_rows= "dynamic")
    else:
        name_insured = st.text_input("Name under which business is conducted: (‘You’)", value = str(response_text_1), key="name_insured")
        address = st.text_input("Addresses of all of your offices & percentage of total fees in each", value = str(response_text_2), key="address")
        date = st.date_input("Date Commenced", value = datetime.strptime(str(response_text_3), "%Y-%m-%d").date(), key="date")
        st.markdown("<p style='font-size:14px; color:black;'>Give details below of all Principals (including details of sole principal)</p>", unsafe_allow_html=True)
        st.data_editor(updated_df, column_config = config,  num_rows= "dynamic")

if __name__ == "__main__":
    main()
