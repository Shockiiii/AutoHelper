import streamlit as st
import os
from backend import query_repair_documents, identify_part_from_image

st.title("Car Repair Assistant (GPT-4 Powered)")

st.markdown("This system has preloaded multiple PDF repair manuals in the background.")

car_model = st.text_input("Enter your car model (e.g., 2020 Toyota Camry)")
user_question = st.text_input("Describe your car issue (e.g., Engine won't start)")

st.markdown("**Optional:** You can either take a photo of the car part using your camera or upload an image file.")


camera_image = st.camera_input("Take a picture")


uploaded_image = st.file_uploader("Or upload a car part image file", type=["jpg", "jpeg", "png"])

if st.button("Find Repair Guide"):
    if not car_model or not user_question:
        st.warning("Please enter both the car model and issue description.")
    else:
        identified_part_info = ""

        
        if camera_image is not None:
            image_data = camera_image.getvalue()
            st.markdown("**Identifying part from camera image...**")
            identified_part_info = identify_part_from_image(car_model, image_data)
            st.markdown(f"**Identified Part (from Image):** {identified_part_info}")
        
        elif uploaded_image is not None:
            image_data = uploaded_image.read()
            st.markdown("**Identifying part from uploaded image...**")
            identified_part_info = identify_part_from_image(car_model, image_data)
            st.markdown(f"**Identified Part (from Image):** {identified_part_info}")
        # skip

        
        result = query_repair_documents(car_model, user_question)
        answer = result["answer"]
        relevant_page = result["relevant_page"]

        st.markdown(f"### Answer:\n\n{answer}")

        if relevant_page is not None:
            st.markdown(f"**Relevant Page:** {relevant_page['pdf_file']} Page {relevant_page['page_number']}")
            image_path = relevant_page.get('image_path')
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption=f"{relevant_page['pdf_file']} - page {relevant_page['page_number']}")
            else:
                st.warning("Page image not found.")
        else:
            st.info("No relevant document found. Answer based on the knowledge base only.")
