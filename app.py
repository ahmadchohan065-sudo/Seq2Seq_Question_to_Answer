import streamlit as st
import torch
import sentencepiece as spm

# reuse your exact classes from earlier days -- copy them into a shared model.py
# and import from there instead of redefining, to avoid two different copies drifting apart
from model import Encoder, lstm_decoder, Seq2Seq
from decode import greedy_search, beam_search  # your Day 3 functions, saved into decode.py

PAD, UNK, BOS, EOS = 0, 1, 2, 3

st.set_page_config(page_title="Urdu Question Generator", layout="wide")


@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file="ur_sp.model")

    encoder = Encoder(vocab_size=8000, emb_dim=256, hidden_dim=512)
    decoder = lstm_decoder(vocab_size=8000, emb_dim=256, dec_dimension=512, enc_dimension=1024)
    model = Seq2Seq(encoder=encoder, decoder=decoder).to(device)

    checkpoint = torch.load("checkpoints/best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, sp, device


model, sp, device = load_everything()

st.title("Urdu Question Generation")
st.write("Paste an Urdu sentence and mark the answer span — the model will generate a question for it.")

full_sentence = st.text_area("Urdu sentence (context)", height=100)
answer_span = st.text_input("Answer span (must appear exactly as written in the sentence above)")

if st.button("Generate Question"):
    if not full_sentence.strip() or not answer_span.strip():
        st.warning("Please fill in both the sentence and the answer span.")
    elif answer_span not in full_sentence:
        st.error("That answer span doesn't appear exactly in the sentence — check spelling/spacing.")
    else:
        # wrap the answer span in <ans>...</ans>, same formatting your model was trained on
        wrapped_input = full_sentence.replace(answer_span, f"<ans> {answer_span} </ans>")

        greedy_output = greedy_search(model, sp, wrapped_input)
        beam_output = beam_search(model, sp, wrapped_input, beam_width=3)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Greedy Output")
            st.write(greedy_output)
        with col2:
            st.subheader("Beam Search Output (k=3)")
            st.write(beam_output)