# load all the necessary libraries
#____________________________________ Step 1 _____________________________________________________
# stramlit for the web app interface 
import streamlit as st

# llm_chains for the language model chains 
# load_normal_chain and are the functions that load the language model chains
from llm_chains import load_normal_chain

# mic_recorder is a function that records the audio from the user
from streamlit_mic_recorder import mic_recorder

# get_timestamp, load_config, get_avatar are functions that are used to get the timestamp, load the configuration file, and get the avatar of the user
from utils import get_timestamp, load_config, get_avatar

# handle_image is a function that handles the image that the user uploads
from image_handler import handle_image

# transcribe_audio is a function that transcribes the audio that the user uploads
from audio_handler import transcribe_audio


# css is the css code that is used to style the chat messages
from html_templates import css

# load_last_k_text_messages, save_text_message, save_image_message, save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history are functions that are used to load the last k text messages, save the text messages, save the image messages, save the audio messages, load the messages, get all the chat history ids, and delete the chat history
from database_operations import load_last_k_text_messages, save_text_message, save_image_message, save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history

# sqlite3 is used to connect to the database
import sqlite3

# load_config is used to load the configuration file
config = load_config()

# _______________________________________Step 2____________________________________________________
# create the helpers functions

# load_chain is a function that loads the language model chain
# and st.cache_resource is a decorator that caches the resource
@st.cache_resource
def load_chain():
    return load_normal_chain()

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def main():

    # the title of our website
    st.title("Hassane Voiceboot 👆")
    st.write(css, unsafe_allow_html=True)

    
    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.audio_uploader_key = 0
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    pdf_toggle_col, voice_rec_col = st.sidebar.columns(2)
    pdf_toggle_col.toggle("PDF Chat", key="pdf_chat", value=False, on_change=clear_cache)
    with voice_rec_col:
        voice_recording=mic_recorder(start_prompt="Record Audio",stop_prompt="Stop recording", just_once=True)
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)
    clear_cache_col.button("Clear Cache", on_click=clear_cache)
    
    chat_container = st.container()

    # here the user will enter his mmessages
    user_input = st.chat_input("Type your message here 👆", key="user_input") 
    
    
    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"], key=st.session_state.audio_uploader_key)
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])


    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input = "Summarize this text: " + transcribed_audio, chat_history=[])
        save_audio_message(get_session_key(), "human", uploaded_audio.getvalue())
        save_text_message(get_session_key(), "ai", llm_answer)
        st.session_state.audio_uploader_key += 2

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input = transcribed_audio, 
                                   chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
        save_audio_message(get_session_key(), "human", voice_recording["bytes"])
        save_text_message(get_session_key(), "ai", llm_answer)

    
    if user_input:
        if uploaded_image:
            with st.spinner("Processing image..."):
                llm_answer = handle_image(uploaded_image.getvalue(), user_input)
                save_text_message(get_session_key(), "human", user_input)
                save_image_message(get_session_key(), "human", uploaded_image.getvalue())
                save_text_message(get_session_key(), "ai", llm_answer)
                user_input = None


        if user_input:
            llm_chain = load_chain()
            llm_answer = llm_chain.run(user_input = user_input, 
                                       chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
            save_text_message(get_session_key(), "human", user_input)
            save_text_message(get_session_key(), "ai", llm_answer)
            user_input = None


    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_messages = load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    if message["message_type"] == "image":
                        st.image(message["content"])
                    if message["message_type"] == "audio":
                        st.audio(message["content"], format="audio/wav")

        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key != None):
            st.rerun()
    # applay the setup_ui 

if __name__ == "__main__":
    main()
