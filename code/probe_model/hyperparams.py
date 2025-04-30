import torch

#Model hyperparameters
model_name = "meta-llama/Llama-3.2-1B"
dtype = torch.bfloat16
model_args = {"torch_dtype" : dtype, "device_map" : "auto"}


#Tokenizer hyperparameters
special_tokens_to_add = {
    "pad_token" : "<|pad|>",
    "additional_special_tokens" : [
        "<|user|>", 
        "<|/user|>", 
        "<|output|>", 
        "<|/output|>", 
    ]
}
embedding_scaler = .03


#Checkpoint hyperparameters
checkpoint_location = "../../checkpoints/sft.pth"


#Prompt hyperparameters
prompts = ["<|user|>Write me a function in Python that intputs a list and outputs the maximum values in the list<|/user|><|output|>"]

#Generation hyperparameters
generation_args = {
    "max_new_tokens" : 250
}