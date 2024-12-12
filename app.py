# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from itertools import groupby
from operator import itemgetter

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

if is_tf_available():
    import tensorflow as tf
    from backend.tensorflow import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    if any(tf.config.experimental.list_physical_devices("gpu")):
        forward_device = tf.device("/gpu:0")
    else:
        forward_device = tf.device("/cpu:0")

else:
    import torch
    from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(det_archs, reco_archs):
    """Build a streamlit layout"""
    # Wide mode
    st.set_page_config(layout="wide")
    
    # Add custom CSS to reduce margins/padding
    st.markdown("""
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 0rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            .element-container {
                margin-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # File selection
    st.markdown("**Document selection**")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"], accept_multiple_files=True)
    
    # Create two main columns for the layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("**Input Document Pages**")
    
    with right_col:
        st.markdown("**Extracted Text**")

    # Move model selection before file handling
    # Model selection
    st.sidebar.title("Model selection")
    st.sidebar.markdown("**Backend**: " + ("TensorFlow" if is_tf_available() else "PyTorch"))
    det_arch = st.sidebar.selectbox("Text detection model", det_archs)
    reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)

    # Parameters
    st.sidebar.title("Parameters")
    assume_straight_pages = st.sidebar.checkbox("Assume straight pages", value=True)
    disable_page_orientation = st.sidebar.checkbox("Disable page orientation detection", value=False)
    disable_crop_orientation = st.sidebar.checkbox("Disable crop orientation detection", value=False)
    straighten_pages = st.sidebar.checkbox("Straighten pages", value=False)
    export_straight_boxes = st.sidebar.checkbox("Export as straight boxes", value=False)
    merge_lines = st.sidebar.checkbox("Merge text by lines", value=True)
    merge_paragraphs = st.sidebar.checkbox("Merge text into paragraphs", value=False)
    line_height_threshold = st.sidebar.slider("Line merge threshold", min_value=0.1, max_value=5.0, value=0.5, step=0.1, 
                                            help="Higher values will merge text that is further apart vertically")
    paragraph_threshold = st.sidebar.slider("Paragraph merge threshold", min_value=1.0, max_value=10.0, value=2.0, step=0.1,
                                          help="Higher values will merge more lines into paragraphs")
    st.sidebar.write("\n")
    bin_thresh = st.sidebar.slider("Binarization threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    st.sidebar.write("\n")
    box_thresh = st.sidebar.slider("Box threshold", min_value=0.1, max_value=0.9, value=0.1, step=0.1)
    st.sidebar.write("\n")



    if uploaded_files:
        # Create predictor once for all files
        with st.spinner("Loading model..."):
            predictor = load_predictor(
                det_arch=det_arch,
                reco_arch=reco_arch,
                assume_straight_pages=assume_straight_pages,
                straighten_pages=straighten_pages,
                export_as_straight_boxes=export_straight_boxes,
                disable_page_orientation=disable_page_orientation,
                disable_crop_orientation=disable_crop_orientation,
                bin_thresh=bin_thresh,
                box_thresh=box_thresh,
                device=forward_device,
            )

        # Process all uploaded files
        all_docs = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".pdf"):
                doc = DocumentFile.from_pdf(uploaded_file.read())
            else:
                doc = DocumentFile.from_images(uploaded_file.read())
            all_docs.extend(doc)

        # Process all pages from all documents
        with st.spinner("Analyzing all pages..."):
            out = predictor(all_docs)
            
            # Create tabs for pages in left column
            with left_col:
                tabs = st.tabs([f"Page {i+1}" for i in range(len(all_docs))])
                
                for idx, (page, tab) in enumerate(zip(all_docs, tabs)):
                    with tab:
                        st.image(page)
            
            # Display complete text in right column
            with right_col:
                all_pages_text = []
                for idx, result in enumerate(out.pages):
                    page_text = []
                    if merge_lines:
                        # Collect all words with their positions
                        words_with_pos = []
                        for block in result.export()['blocks']:
                            for line in block['lines']:
                                for word in line['words']:
                                    # Get word geometry
                                    x_min, y_min = word['geometry'][0]
                                    x_max, y_max = word['geometry'][1]
                                    center_y = (y_min + y_max) / 2
                                    height = y_max - y_min
                                    words_with_pos.append({
                                        'text': word['value'],
                                        'center_y': center_y,
                                        'x_min': x_min,
                                        'height': height
                                    })
                        
                        # Sort words by vertical position
                        words_with_pos.sort(key=lambda x: x['center_y'])
                        
                        # Group words into lines based on vertical position
                        current_line = []
                        lines = []
                        if words_with_pos:
                            current_line = [words_with_pos[0]]
                            avg_height = words_with_pos[0]['height']
                            
                            for word in words_with_pos[1:]:
                                # Check if word belongs to current line
                                prev_center = current_line[-1]['center_y']
                                curr_center = word['center_y']
                                threshold = avg_height * line_height_threshold
                                
                                if abs(curr_center - prev_center) <= threshold:
                                    current_line.append(word)
                                    # Update average height
                                    avg_height = sum(w['height'] for w in current_line) / len(current_line)
                                else:
                                    # Sort words in current line by x position
                                    current_line.sort(key=lambda x: x['x_min'])
                                    lines.append([w['text'] for w in current_line])
                                    current_line = [word]
                                    avg_height = word['height']
                            
                            # Add the last line
                            if current_line:
                                current_line.sort(key=lambda x: x['x_min'])
                                lines.append([w['text'] for w in current_line])
                        
                        # Join words in each line
                        page_text = [' '.join(line) for line in lines]
                        
                        # Merge lines into paragraphs if enabled
                        if merge_paragraphs and page_text:
                            paragraphs = []
                            current_paragraph = [page_text[0]]
                            
                            for i in range(1, len(page_text)):
                                current_line = page_text[i]
                                prev_line = page_text[i-1]
                                
                                # Calculate the vertical distance between lines
                                prev_words = [w for w in words_with_pos if w['text'] in prev_line.split()]
                                curr_words = [w for w in words_with_pos if w['text'] in current_line.split()]
                                
                                if prev_words and curr_words:
                                    prev_bottom = max(w['center_y'] + w['height']/2 for w in prev_words)
                                    curr_top = min(w['center_y'] - w['height']/2 for w in curr_words)
                                    avg_height = (sum(w['height'] for w in prev_words) + sum(w['height'] for w in curr_words)) / (len(prev_words) + len(curr_words))
                                    
                                    # If the gap between lines is small enough, merge into paragraph
                                    if (curr_top - prev_bottom) <= (avg_height * paragraph_threshold):
                                        current_paragraph.append(current_line)
                                    else:
                                        paragraphs.append(' '.join(current_paragraph))
                                        current_paragraph = [current_line]
                                else:
                                    current_paragraph.append(current_line)
                            
                            # Add the last paragraph
                            if current_paragraph:
                                paragraphs.append(' '.join(current_paragraph))
                            
                            page_text = paragraphs
                    else:
                        for block in result.export()['blocks']:
                            block_text = []
                            for line in block['lines']:
                                line_words = []
                                for word in line['words']:
                                    line_words.append(word['value'])
                                block_text.append(' '.join(line_words))
                            page_text.append('\n'.join(block_text))
                    
                    all_pages_text.append(f"--- Page {idx + 1} ---\n{chr(10).join(page_text)}\n")
                
                complete_text = '\n'.join(all_pages_text)
                st.text_area("", complete_text, height=900)
                
                # Add download button for the text
                st.download_button(
                    label="Download extracted text",
                    data=complete_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main(DET_ARCHS, RECO_ARCHS)
