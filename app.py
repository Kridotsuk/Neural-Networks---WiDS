import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = "./dialogpt-finetuned/final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- 2. SESSION STATE (Replacing the 'While' loop variables) ---
if "chat_history_ids" not in st.session_state:
    # Initialize as an empty tensor just like your working fix
    st.session_state.chat_history_ids = torch.tensor([], dtype=torch.long)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. UI DISPLAY ---
st.title("🎬 Movie Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. THE CORE LOGIC (Modeled on your working block) ---
if prompt := st.chat_input("Write a movie line..."):
    # Display & Store User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Step 1: Encode Input
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # Step 2: History & Truncate (The "Secret Sauce")
    if st.session_state.chat_history_ids.shape[-1] > 0:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids
    
    if bot_input_ids.shape[-1] > 256:
        bot_input_ids = bot_input_ids[:, -256:]

    # Step 3: Attention Mask
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Step 4: GENERATE (The Exact Working Params)
    with st.spinner("Recording..."):
        output_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            min_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.5
        )

    # Step 5: Slicing & Decoding
    response_ids = output_ids[:, bot_input_ids.shape[-1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Step 6: UI Output & Fallback
    if not response.strip():
        response = "(The character stares back in silence...)"
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Step 7: Update Persistent History
    st.session_state.chat_history_ids = output_ids
