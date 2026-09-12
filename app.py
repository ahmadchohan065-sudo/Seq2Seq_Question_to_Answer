import streamlit as st
import torch
import sentencepiece as spm

from model import Encoder, lstm_decoder, Seq2Seq,greedy_search, beam_search,generate_question

st.markdown("""
<style>
.stApp {
    background-color: white;
}
h1 {
    text-align: center;
    color: #1E90FF !important;
}
h3 {
    color: #333333 !important;
}
.stApp p, .stMarkdown, .stMarkdown p {
    text-align: center;
    color: #333333 !important;
}
div.stButton {
    display: flex;
    justify-content: center;
}
div.stButton button {
    background-color: #FFD700;
    color: black;
    border-radius: 8px;
}
label {
    color: #333333 !important;
}
textarea, input[type="text"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #1E90FF !important;
}
</style>
""", unsafe_allow_html=True)

PAD, UNK, BOS, EOS = 0, 1, 2, 3

st.set_page_config(page_title="Urdu Question Generator", layout="wide")


@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file="ur_sp.model")

    encoder = Encoder(vocab_size=8000, emb_dim=256, hidden_dim=512)
    decoder = lstm_decoder(vocab_size=8000, emb_dim=256, dec_dimension=512, enc_dimension=1024)
    model = Seq2Seq(encoder=encoder, decoder=decoder).to(device)

    checkpoint = torch.load("checkpoint-last-last/best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, sp, device


model, sp, device = load_everything()

st.title("Urdu Question Generation")
st.markdown("<p style='font-size:20px; font-weight:bold;'>Paste an Urdu sentence and mark the answer span — the model will generate a question for it.</p>", unsafe_allow_html=True)

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

        greedy_output, beam_output = generate_question(wrapped_input, sp, model)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Greedy Output")
            st.markdown(f"""
            <div style='background-color:#F0F8FF; padding:15px; border-radius:10px; border:1px solid #1E90FF; text-align:center;'>
            <p style='font-size:22px; color:black; font-weight:bold; margin:0;'>{greedy_output}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("Beam Search Output (k=3)")
            st.markdown(f"""
            <div style='background-color:#FFFDE7; padding:15px; border-radius:10px; border:1px solid #FFD700; text-align:center;'>
            <p style='font-size:22px; color:black; font-weight:bold; margin:0;'>{beam_output}</p>
            </div>
            """, unsafe_allow_html=True)