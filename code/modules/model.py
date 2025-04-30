import torch


#Add the special tokens to the tokenizer, expand the embedding table of the model, initialize the token embeddings
def add_special_tokens(tokenizer, model, special_tokens_to_add, embedding_scaler):
    #Add the special tokens to the tokenizer and resize the embedding table of the model
    tokenizer.add_special_tokens(special_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer), mean_resizing = False)   #We will do mean_resizing = False because we are going to override the intialization weights anyway
    
    #Set the initial embeddings of the added tokens that are NOT the padding token to be like eos_token or bos_token plus some random noise
    with torch.no_grad():
        for token in special_tokens_to_add["additional_special_tokens"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if "/" in token:
                current_id = tokenizer.eos_token_id
            else:
                current_id = tokenizer.bos_token_id
            current_embedding = model.base_model.embed_tokens.weight[current_id].clone()
            model.base_model.embed_tokens.weight[token_id] = current_embedding + embedding_scaler*torch.randn_like(current_embedding)

    #Set the initial embeddings of the pad token to be 0
    with torch.no_grad():
        model.base_model.embed_tokens.weight[tokenizer.pad_token_id] = torch.zeros_like(model.base_model.embed_tokens.weight[tokenizer.pad_token_id])

    #Use .register_hooks() to ensure that the pad token is not updated during training
    def zero_pad_grad(pad_id):
        def hook(grad):
            grad[pad_id] = torch.zeros_like(grad[pad_id])
            return grad
        return hook
    model.base_model.embed_tokens.weight.register_hook(zero_pad_grad(tokenizer.pad_token_id))




