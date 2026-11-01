import streamlit as st
from services.persistence.exercise_repository import get_or_create_user




def render_login_page():
    if st.session_state.get("user_id") is not None:
        return True


    st.title("️🏋️ AI Real-Time GYM Coach 💪")
    st.markdown("###Welcome ! Please enter your credentials to access the application.")


    with st.form("Login_form" , clear_on_submit=False):
        username = st.text_input("Name (unique)" , placeholder= "unique name e.g siya")
        submit_button = st.form_submit_button("Start session", width="stretch")

    if submit_button:
        if not username:
            st.error("Please enter a valid username.")  
            return False


        user = get_or_create_user(username)


        st.session_state["user_id"] = user["id"]
        st.session_state["username"] = user["username"]


        st.rerun()

    return False

