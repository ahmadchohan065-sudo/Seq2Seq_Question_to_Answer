import torch


def greedy_search(model,src,src_lens):
    model.eval()
    with torch.no_grad():
        encoder_outputs,hidden_state,cell_state=model.encoder(src,src_lens)
        mask = make_src_mask(src_lens.to(src.device), src.size(1))
        dec_hidden=torch.stack([hidden_state,hidden_state],dim=0)
        dec_cell=torch.stack([cell_state,cell_state],dim=0)
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
        encoder_outputs, hidden_state, cell_state = model.encoder(src, src_lens)
        mask = make_src_mask(src_lens.to(src.device), src.size(1))
        dec_hidden = torch.stack([hidden_state, hidden_state], dim=0)
        dec_cell = torch.stack([cell_state, cell_state], dim=0)
        dec_states = (dec_hidden, dec_cell)

        beam = [([BOS], 0.0, dec_states)]
        completed_beams = []

        for i in range(25):
            all_candidates = []

            for tokens, score, state in beam:
                if tokens[-1] == EOS:
                    completed_beams.append([tokens, score])
                    continue

                current_token = torch.tensor([[tokens[-1]]], device=device)

                output_scores, new_state, alpha = model.decoder(
                    current_token, state, encoder_outputs,mask
                )

                output_scores = torch.log_softmax(output_scores, dim=1)
                penalty = 1.5
                for t in set(tokens):
                    if t not in (BOS, EOS, PAD):
                        output_scores[0, t] -= penalty

                top_scores, top_token = output_scores.topk(
                    beam_size, dim=1
                )

                for j in range(beam_size):
                    new_token = top_token[0, j].item()
                    new_score = score + top_scores[0, j].item()

                    all_candidates.append(
                        (tokens + [new_token], new_score, new_state)
                    )

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[:beam_size]

        all_completed = completed_beams.copy()

        for tokens, score, _ in beam:
            all_completed.append((tokens, score))

        all_completed.sort(key=lambda x: x[1], reverse=True)

        best_tokens = all_completed[0][0]

        result = []

        for t in best_tokens:
            if t != BOS and t != EOS and t != PAD:
                result.append(t)

        return result


