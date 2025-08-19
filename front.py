import streamlit as st

from rag import query_about_nutrition
from rag import chat_history


st.header("_SmartFit:_ is :blue[AI-Powered Fitness Coach] :sunglasses:")
 
question = st.text_input("Enter Your query about nutrition")

if st.button("Submit"):
    st.warning("Please wait, we are retriving you answer!")
    ans = query_about_nutrition(question)
    st.write(ans)
    st.success("Done!")

st.subheader("Chat History", divider=True)
for key,value in enumerate(chat_history):
    st.text(f"{key}:{value}")