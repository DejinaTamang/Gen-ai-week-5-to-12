import streamlit as st

st.title("Week 8 - Document Q&A")

st.write("Streamlit is working!")

name = st.text_input("Enter your question")

if st.button("Submit"):
    st.success(f"You entered: {name}")