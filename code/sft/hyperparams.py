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
output_token = "<|output|>"

#Dataset / DataLoader hyperparameters
dataset_name = "sahil2801/CodeAlpaca-20k"
test_size = 0.15
batch_size = 16
pin_memory = True 
num_workers = 0

#lr_scheduler hyperparameters
scheduler_patience = 2      #How many validation loops with no decrease in validation loss before the learning rate is multiplied by factor
scheduler_factor = .1       #The factor that the learning rate gets multiplied by when it is not improving

#Training loop hyperparameters
epochs = 15
initial_val = True
val_interval = 1
smallest_val_loss = float("inf")
stop_patience = 3          #How many validation loops with no decrease in validation loss before the training stops

#Checkpoint hyperparameters
checkpoint_storage_location = "../../checkpoints/sft.pth"
graph_save_location = "../../outputs/graphs"