import streamlit as st
import requests
import json

st.set_page_config(page_title="GuguGaga Chat", page_icon="🤖")
st.title("🤖 GuguGaga AI Assistant")

# 侧边栏配置
with st.sidebar:
    st.header("参数设置")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.85)
    top_p = st.slider("Top P", 0.0, 1.0, 0.85)
    max_tokens = st.number_input("Max New Tokens", 128, 8192, 2048)
    
    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("说点什么..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 调用 API
            api_url = "http://localhost:8000/chat"
            payload = {
                "messages": st.session_state.messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_tokens
            }
            
            with requests.post(api_url, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            st.error(f"连接后端失败: {e}")

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})