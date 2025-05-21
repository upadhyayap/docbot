from dotenv import load_dotenv
from backend.core import run_llm
import streamlit as st
import time
from typing import Set
import pathlib

load_dotenv()

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

st.set_page_config(page_title="DocBot", page_icon=":robot:")

# Inject custom CSS for neon theme and modern look
def set_custom_css():
    css_path = pathlib.Path("custom_theme.css")
    if css_path.exists():
        with open(css_path) as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

set_custom_css()

# Add user profile information in the sidebar
with st.sidebar:
    st.title("User Profile")
    
    # Add a profile picture (using an emoji as placeholder)
    st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", width=100)
    
    # Add user information
    st.subheader("John Doe")  # Replace with actual user name
    st.caption("john.doe@example.com")  # Replace with actual email
    
    # Add a divider
    st.divider()
    
    # Add some stats or additional information
    st.write("**Chat Stats**")
    st.write(f"Total Messages: {len(st.session_state.get('chat_history', []))}")
    
    # Add a logout button
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

st.header("Ask Anything About LangChain")
prompt = st.text_input("prompt", placeholder="Enter a prompt here...")

if (
    "user_prompt_history" not in st.session_state
    and "chat_answer_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state.user_prompt_history = []
    st.session_state.chat_answer_history = []
    st.session_state.chat_history = []


if prompt:
    with st.spinner("Generating response..."):
        time.sleep(2)
        result = run_llm(query=prompt, chat_history=st.session_state.chat_history)
        generated_response = result

        sources = set(doc.metadata["source"] for doc in generated_response["source_documents"])
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answer_history.append(formatted_response)
        st.session_state.chat_history.append(("human", prompt))
        st.session_state.chat_history.append(("ai", generated_response["result"]))

if st.session_state.chat_answer_history:
    for i, (prompt, answer) in enumerate(zip(st.session_state.user_prompt_history, st.session_state.chat_answer_history)):
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write(answer)

st.divider()
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: #666;'>
        Made with ❤️ using LangChain and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)




    

