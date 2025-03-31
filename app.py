import streamlit as st
from frontend.home_page import show_home_page
from frontend.result_page import show_result_page

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Results"])

    if "navigate_to_results" not in st.session_state:
        st.session_state["navigate_to_results"] = False

    if page == "Home" and not st.session_state["navigate_to_results"]:
        show_home_page()
    elif page == "Results" or st.session_state["navigate_to_results"]:
        show_result_page()
        st.session_state["navigate_to_results"] = False  # Reset after showing results

if __name__ == "__main__":
    main()
