import streamlit as st

from rag import query_about_nutrition


question = st.text_input("Enter Your query about nutrition")


if st.button("Submit"):
    st.warning("Please wait, we are retriving you answer!")
    ans = query_about_nutrition(question)
    st.write(ans)
    st.success("Done!")