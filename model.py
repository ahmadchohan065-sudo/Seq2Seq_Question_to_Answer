import random
import torch
import torch.nn as nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3

class Encoder(nn.Module):
    def __init__(self, vocab_size=8000, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.3)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)

        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden_proj = torch.tanh(self.fc_hidden(hidden_cat))
        cell_proj = torch.tanh(self.fc_cell(cell_cat))

        return outputs, hidden_proj, cell_proj

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
        batch_size = src.shape[0]
        target_len = target.shape[1]
        vocab_size=8000
        output=(batch_size,target_len,vocab_size)
        encoder_outputs,hidden_state,cell_state=self.encoder(src,src_lens)
        mask = make_src_mask(src_lens, src.size(1))
        dec_hidden=torch.stack([hidden_state,hidden_state],dim=0)
        dec_cell=torch.stack([cell_state,cell_state],dim=0)
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
