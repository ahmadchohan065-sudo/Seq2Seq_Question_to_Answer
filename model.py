import random
import torch
import torch.nn as nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3

import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
PAD, UNK, BOS, EOS = 0, 1, 2, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, vocab_size=8000, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.3)

        self.fc_hidden_l0 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell_l0   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_hidden_l1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell_l1   = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=src.size(1))

        h_l0 = torch.cat((hidden[0], hidden[1]), dim=1)
        c_l0 = torch.cat((cell[0], cell[1]), dim=1)
        h_l1 = torch.cat((hidden[2], hidden[3]), dim=1)
        c_l1 = torch.cat((cell[2], cell[3]), dim=1)

        h0 = torch.tanh(self.fc_hidden_l0(h_l0))
        c0 = torch.tanh(self.fc_cell_l0(c_l0))
        h1 = torch.tanh(self.fc_hidden_l1(h_l1))
        c1 = torch.tanh(self.fc_cell_l1(c_l1))

        dec_hidden = torch.stack([h0, h1], dim=0)
        dec_cell   = torch.stack([c0, c1], dim=0)

        return outputs, dec_hidden, dec_cell


def make_src_mask(src_lens, max_len):
    return torch.arange(max_len, device=src_lens.device)[None, :] < src_lens[:, None]


class Bahdanau_Attention(nn.Module):
    def __init__ (self,enc_dimension=1024, dec_dimension=512):
        super().__init__()
        self.w_key=nn.Linear(enc_dimension,dec_dimension,bias=True)
        self.w_query=nn.Linear(dec_dimension,dec_dimension,bias=True)
        self.w_value=nn.Linear(dec_dimension,1,bias=True)

    def forward(self,query,keys,mask):
        query_expanded = query.unsqueeze(1)
        e=self.w_value(torch.tanh(self.w_query(query_expanded)+self.w_key(keys)))
        e = e.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        alpha=torch.softmax(e,dim=1)
        context_vector=alpha*keys
        context_vector=torch.sum(context_vector,1,keepdim=True)
        return context_vector,alpha

class lstm_decoder(nn.Module):
    def __init__(self,vocab_size=8000,emb_dim=256,dec_dimension=512,enc_dimension=1024):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.attention=Bahdanau_Attention(enc_dimension,dec_dimension)
        self.lstm=nn.LSTM(enc_dimension+emb_dim,dec_dimension,num_layers=2,dropout=0.3,batch_first=True,)
        self.out=nn.Linear(dec_dimension + enc_dimension,vocab_size)

    def forward(self,input,dec_state,enc_out,mask):
        embedd=self.embedding(input)
        top_hidden=dec_state[0][-1]
        context_vector,alpha=self.attention(top_hidden,enc_out,mask)
        lstm_input=torch.cat((embedd, context_vector), dim=2)
        lstm_out, new_dec_states = self.lstm(lstm_input, dec_state)
        combined = torch.cat((lstm_out, context_vector), dim=2)
        output_scores=self.out(combined.squeeze(1))

        return output_scores, new_dec_states, alpha


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, target, teacher_ratio=0.5):
        target_len = target.shape[1]
        encoder_outputs,dec_hidden,dec_cell=self.encoder(src,src_lens)
        mask = make_src_mask(src_lens, src.size(1))
        dec_states=(dec_hidden,dec_cell)
        input_token=target[:,0].unsqueeze(1)
        outputs = []

        for t in range(1,target_len):
            output_scores,dec_states,alpha = self.decoder(input_token,dec_states,encoder_outputs,mask)
            outputs.append(output_scores)
            teacher_force = random.random() < teacher_ratio
            top_pred = output_scores.argmax(dim=1).unsqueeze(1)
            input_token = (target[:, t].unsqueeze(1) if teacher_force else top_pred)
        outputs=torch.stack(outputs,dim=1)
        return outputs


def greedy_search(model,src,src_lens):
    model.eval()
    with torch.no_grad():
        encoder_outputs,dec_hidden,dec_cell=model.encoder(src,src_lens)
        mask = make_src_mask(src_lens.to(src.device), src.size(1))
        dec_states=(dec_hidden,dec_cell)
        generated_tokens=[]
        input_token = torch.tensor([BOS]).unsqueeze(0).to(device)
        for i in range(25):
            output_scores,dec_states,alpha=model.decoder(input_token,dec_states,encoder_outputs,mask)
            best_score=output_scores.argmax(dim=1).item()
            if best_score==EOS:
                break
            generated_tokens.append(best_score)
            input_token=torch.tensor([[best_score]],device=device)
    return generated_tokens


def beam_search(model, src, src_lens, beam_size=5):
    model.eval()

    with torch.no_grad():
        encoder_outputs,dec_hidden,dec_cell=model.encoder(src,src_lens)
        mask = make_src_mask(src_lens.to(src.device), src.size(1))
        dec_states=(dec_hidden,dec_cell)
        beams = [([BOS], 0.0, dec_states)]
        finished = []

        for step in range(25):
            candidates = []

            for tokens, score, state in beams:
                if tokens[-1] == EOS:
                    finished.append((tokens, score))
                    continue

                last_token = torch.tensor([[tokens[-1]]], device=device)
                output_scores, new_state, alpha = model.decoder(last_token, state, encoder_outputs, mask)
                output_scores = torch.log_softmax(output_scores, dim=1)

                top_scores, top_tokens = output_scores.topk(beam_size, dim=1)

                for i in range(beam_size):
                    new_tokens = tokens + [top_tokens[0, i].item()]
                    new_score = score + top_scores[0, i].item()
                    candidates.append((new_tokens, new_score, new_state))
            def avg_score(c):
                return c[1] / len(c[0])

            candidates.sort(key=avg_score, reverse=True)
            beams = candidates[:beam_size]

        for tokens, score, state in beams:
            finished.append((tokens, score))

        finished.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        best_tokens = finished[0][0]

        result = []
        for t in best_tokens:
            if t not in (BOS, EOS, PAD):
                result.append(t)

        return result

def generate_question(raw_text, sp_model, model):
    device = next(model.parameters()).device
    src_ids = sp_model.encode(raw_text)

    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_lens = torch.tensor([len(src_ids)])

    greedy_tokens = greedy_search(model, src_tensor, src_lens)
    greedy_output = sp_model.decode(greedy_tokens)

    beam_tokens = beam_search(model, src_tensor, src_lens, beam_size=3)
    beam_output = sp_model.decode(beam_tokens)
    return greedy_output, beam_output

