import streamlit as st
from backend import query_repair_documents
import os

st.title("Car Repair Assistant (GPT-4 Powered)")

st.markdown("This system has preloaded multiple PDF repair manuals in the background. When you enter your car model and describe the issue, the system will retrieve the most relevant pages from these manuals, displaying the original page (including images and formatting) along with an answer based on the document.")

car_model = st.text_input("Enter your car model (e.g., 2020 Toyota Camry)")
user_question = st.text_input("Describe your car issue (e.g., Engine won't start)")

if st.button("Find Repair Guide"):
    if not car_model or not user_question:
        st.warning("Please enter both the car model and issue description.")
    else:
        result = query_repair_documents(car_model, user_question)
        answer = result["answer"]
        relevant_page = result["relevant_page"]
        
        st.markdown(f"### Answer:\n\n{answer}")
        
        if relevant_page is not None:
            st.markdown(f"**Relevant Page:** {relevant_page['pdf_file']} Page {relevant_page['page_number']}")
            # Display the page image
            image_path = relevant_page['image_path']
            if os.path.exists(image_path):
                st.image(image_path, caption=f"{relevant_page['pdf_file']} - page {relevant_page['page_number']}")
            else:
                st.warning("Page image not found.")
        else:
            st.info("No relevant document found. Answer based on the knowledge base only.")
