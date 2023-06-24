import torch
import clip 


def _apply_clip_text_model(clip_text_model, data, device):
    #data['raw_text'] = [cap for item in data['raw_text'] for cap in item]
    with torch.no_grad():
        #print(len(data['raw_text']))
        #print(type(data["raw_text"]))
        #print(type(clip.tokenize))
        input_x = clip.tokenize(data['raw_text'], truncate=True).to(device)
        #print("input goign to clip",input_x.shape)
        x = clip_text_model.token_embedding(input_x).type(
            clip_text_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + clip_text_model.positional_embedding.type(clip_text_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_text_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = clip_text_model.ln_final(x).type(clip_text_model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ clip_text_model.text_projection

        x = x.detach().cpu()
        input_x = input_x.cpu()

        batch_size, _, dim = x.shape
        prev_n_tokens = data['text'].shape[1]

        input_x = input_x[:, 1:]  # first token is a token of beginning of the sentence
        x = x[:, 1:]  # first token is a token of beginning of the sentence

        new_text = x[:, :prev_n_tokens]
        new_text_mask = torch.zeros(batch_size, prev_n_tokens)

        for i in range(len(input_x)):
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            n_eot = input_x[i].argmax().item()
            new_text_mask[i, :n_eot] = 1

        data['text'] = new_text.type(data['text'].dtype)
        data['text_mask'] = new_text_mask.type(data['text_mask'].dtype)
    #print("data shape",data["text"].shape,data["text_mask"].shape)
    return data
