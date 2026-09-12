# Urdu Question Generation (Seq2Seq, built from scratch)

This project is a sequence-to-sequence model that reads an Urdu sentence with an answer marked inside it, and generates the question that answer responds to.

Example (translated to English for clarity):
**Input:** "The Indus is about `<ans>3,180 km</ans>` long"
**Output:** "How long is the Indus?"

Everything here — the encoder, the decoder, the attention mechanism, the tokenizer — was built and trained from scratch. No pretrained weights, no Transformers, no off-the-shelf seq2seq libraries.

---

## How it works

We use an RNN encoder-decoder with attention:

- **Encoder:** 2-layer bidirectional LSTM
- **Decoder:** 2-layer LSTM with Bahdanau attention
- **Embedding size:** 256
- **Hidden size:** 512
- **Vocabulary:** 8,000 subword tokens, learned with SentencePiece on our own training data
- **Decoding:** both greedy search and beam search (k=3) are supported

The model was trained with teacher forcing (gradually reduced over epochs) and cross-entropy loss that ignores padding tokens.

---

## Dataset

We used [UQA](https://huggingface.co/datasets/uqa/UQA) — a Corpus for Urdu Question Answering that translates SQuAD 2.0 into Urdu while keeping the answer-span offsets intact. Around 142k context-question-answer rows, though we drop the unanswerable ones.

Since a from-scratch model on this much data can't realistically read a full 300-token paragraph, we followed the same setup as Du et al. (2017) and only fed the model the single sentence that actually contains the answer. That's what keeps the task learnable at this scale.

For out-of-domain testing, we also evaluate on [Wiki-UQA](https://huggingface.co/datasets/uqa/Wiki-UQA).

---

## Project structure

```
.
├── app.py                  # Streamlit front-end
├── model.py                 # Encoder, decoder, attention, and the model class
├── train.tsv / valid.tsv    # sentence-answer-question pairs after preprocessing
├── ur_sp.model               # trained SentencePiece tokenizer
├── checkpoints/               # saved model checkpoints (not pushed to GitHub — see below)
├── results/
│   ├── samples.tsv            # source, reference, greedy output, beam output for 50 examples
│   └── human_eval.csv         # human evaluation scores from both team members
└── README.md
```

## Setting it up

```bash
pip install torch sentencepiece streamlit datasets sacrebleu rouge-score
```

## Running the front end

```bash
streamlit run app.py
```

This opens a simple web page where you can paste an Urdu sentence, mark the answer span, and see both the greedy and beam search outputs side by side.

## Reproducing training

If you want to retrain from scratch, the data prep, tokenizer training, model, and training loop are all in `seq2seq_model.ipynb`. It's meant to be run on Colab or Kaggle with a free GPU — training takes roughly 1-2 hours depending on how many epochs you go for.

---

## Results

**Model**

| | |
|---|---|
| Encoder / Decoder | 2-layer BiLSTM / 2-layer LSTM + attention |
| Vocabulary size | 8,000 |
| Trainable parameters |34508097 |
| Optimizer | Adam |

**Automatic metrics**

| Split | Decoding | BLEU-4 | ROUGE-L | Perplexity | `<unk>` rate |
|---|---|---|---|---|---|
| UQA validation | greedy |4.505743088900178 |0.0096664099858 |759.3 |0.007531645569620 |
| UQA validation | beam (k=3) |5.461791188666278 | 759.3|0.010783922273 |0.008311445327 |
| Wiki-UQA | greedy |3.274277019330637 | 0.0 |759.3 |0.02536231884057971 |
| Wiki-UQA | beam (k=3) |3.0873385683127896 |759.3 |0.00564971751412429 | 0.0244470314 |

**Human evaluation (50 samples)**

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 (% yes) |48 |84 |56 |
| Member 2 (% yes) |66 | 88|64 |
| Cohen's κ |0.64 |0.66|0.66 |

A few qualitative examples (good and bad) along with the full results are in `results/samples.tsv`.

---

## Team

- **[Name 1]** — [[musfiraumar](https://github.com/musfiraumar)]
- **[Name 2]** — []

## Links

- Medium blog: [link]
- LinkedIn post: [link]

## Acknowledgments

- Dataset: Arif, Farid, Athar and Raza — *UQA: Corpus for Urdu Question Answering*, LREC-COLING 2024
- Task setup inspired by Du et al. (2017), *Learning to Ask: Neural Question Generation for Reading Comprehension*
