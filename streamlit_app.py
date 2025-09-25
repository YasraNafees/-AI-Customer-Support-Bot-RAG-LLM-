import streamlit as st
import requests

st.set_page_config(page_title="AI Customer Support Bot", layout="centered")

# Chat history state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = [{
        "role": "assistant",
        "content": "👋 Hi! I'm your AI Customer Support Bot. Ask me anything about your support tickets!"
    }]

# Display chat history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your question...")

if user_input:
    # Save and display user input
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call FastAPI
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.get("http://127.0.0.1:8000/ask", params={"query": user_input})
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                else:
                    answer = f"❌ API error {response.status_code}: {response.text}"
            st.markdown(answer)

        st.session_state["message_history"].append({"role": "assistant", "content": answer})

    except requests.exceptions.ConnectionError:
        error_msg = "🚫 Cannot connect to FastAPI. Is it running at http://127.0.0.1:8000?"
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.session_state["message_history"].append({"role": "assistant", "content": error_msg})
    except Exception as e:
        error_msg = f"⚠️ Error: {e}"
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.session_state["message_history"].append({"role": "assistant", "content": error_msg})
