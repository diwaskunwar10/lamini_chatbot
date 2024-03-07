import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pre-trained model and tokenizer
model_name = "lamini/LaMini-Flan-T5-248M"
cache_dir = "/home/diwas/Documents/ai/cache/"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

# Define a function to generate responses
def get_lamini_response(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Lamini Chatbot")

user_input = st.text_input("You:")
if user_input:
    response = get_lamini_response(user_input, max_new_tokens=500)
    st.text_area("Chatbot:", value=response, height=200)

st.sidebar.info("Type 'quit', 'exit', or 'q' to end the conversation.")
