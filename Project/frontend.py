import streamlit as st
import os
from backend import query_repair_documents, identify_part_from_image
from streamlit_chat import message  # 用于显示对话气泡

# 设置页面属性
st.set_page_config(page_title="Car Repair Assistant", layout="wide")
st.title("Car Repair Assistant (GPT-4 Powered)")

# 侧边栏中输入车型
with st.sidebar:
    st.header("车辆信息")
    car_model = st.text_input("请输入车型 (例如：2020 Toyota Camry)")

# 初始化会话状态（首次运行时添加欢迎消息）
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "欢迎使用汽车维修助手，请描述您的问题。"}
    ]

# 聊天输入区：采用 st.form 一次性收集文本和图片输入
with st.form("input_form", clear_on_submit=True):
    user_question = st.text_input("描述您的车辆问题 (例如：发动机无法启动)")
    st.markdown("**可选：** 您可以拍摄或上传车件图片")
    camera_image = st.camera_input("拍摄图片", key="cam_input")
    uploaded_image = st.file_uploader("上传图片文件", type=["jpg", "jpeg", "png"], key="file_input")
    submitted = st.form_submit_button("发送")

# 如果表单提交了，则处理用户输入和调用后端
if submitted:
    if not car_model or not user_question:
        st.warning("请先在侧边栏输入车型以及问题描述。")
    else:
        # 获取图片数据（如果有上传或拍摄）
        image_data = None
        if camera_image is not None:
            image_data = camera_image.getvalue()
        elif uploaded_image is not None:
            image_data = uploaded_image.read()

        # 将用户输入添加到会话记录中
        user_msg = {"role": "user", "content": user_question}
        if image_data:
            user_msg["image"] = image_data
        st.session_state.messages.append(user_msg)

        # 如果有图片，则先调用后端对图片进行零件识别
        if image_data:
            st.info("正在识别图片中的零件...")
            identified_part_info = identify_part_from_image(car_model, image_data)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"图片识别结果：{identified_part_info}"
            })

        # 调用后端查询相关维修文档和生成答案
        result = query_repair_documents(car_model, user_question)
        answer = result.get("answer", "抱歉，未能获取答案。")
        relevant_page = result.get("relevant_page")

        assistant_response = {"role": "assistant", "content": answer}
        if relevant_page:
            pdf_info = f"相关PDF：{relevant_page['pdf_file']} 第 {relevant_page['page_number']} 页"
            assistant_response["content"] += f"\n\n{pdf_info}"
            image_path = relevant_page.get("image_path")
            if image_path and os.path.exists(image_path):
                assistant_response["pdf_image"] = image_path
                assistant_response["pdf_caption"] = f"{relevant_page['pdf_file']} - 第 {relevant_page['page_number']} 页"
            else:
                assistant_response["content"] += "\n\n[警告：未找到PDF页面图片。]"
        else:
            assistant_response["content"] += "\n\n未找到相关文档，答案基于知识库生成。"

        st.session_state.messages.append(assistant_response)

# 在处理完用户输入后，再显示对话记录，这样最新的消息就能立即显示出来
st.markdown("### 对话记录")
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
        # 如果用户消息中包含图片，则显示该图片
        if "image" in msg:
            st.image(msg["image"], caption="用户上传/拍摄的图片")
    else:
        message(msg["content"], is_user=False, key=f"assistant_{i}")
        # 如果回复中包含PDF图片，则显示该图片
        if "pdf_image" in msg:
            st.image(msg["pdf_image"], caption=msg.get("pdf_caption", ""))
