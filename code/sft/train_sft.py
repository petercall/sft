#Regular Imports
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer


#Filepath to add to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#File Imports
import hyperparams as hp
from modules import train, graph_losses, add_special_tokens, return_loaders




#Start Code--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Change to the current working directory to ensure relative imports work properly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Define the model and tokenizer, add in the special tokens, and get the token id of the output token
model = AutoModelForCausalLM.from_pretrained(hp.model_name, **hp.model_args)
tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
add_special_tokens(tokenizer, model, hp.special_tokens_to_add, hp.embedding_scaler)
output_token_id = tokenizer(hp.output_token, add_special_tokens = False)["input_ids"][0]      #This is the token ID of the token that marks when the assistant starts speaking. We need it because we need to mask out in the y tensor all tokens that come before this token, so that we are only performing CELoss on the model's response, and not on the user's input sentence.

#Define the optimizer, lr_shceduler, and loss_function
optimizer = optim.AdamW(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = hp.scheduler_patience, factor = hp.scheduler_factor)
loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)    #Make sure to ignore the pad token id when you are doing SFT so that we aren't back-propogating based on how well the model can predict a padding token!!!

#Get the train and data loader
train_loader, val_loader = return_loaders(hp.dataset_name, tokenizer, output_token_id, hp.test_size, hp.batch_size, hp.pin_memory, hp.num_workers)

# Train the model
train_losses, val_losses, smallest_val_loss = train(
    model,              #model
    optimizer,          #optimizer
    scheduler,          #lr scheduler
    loss_function,      #loss function
    train_loader,       #train loader
    val_loader,         #val loader
    hp.epochs,             #epochs to train for
    hp.initial_val,        #Whether to do get a validation loss before starting training
    hp.val_interval,       #How often to validate
    hp.smallest_val_loss,  #current smallest validation loss
    hp.stop_patience,      #Number of validation runs with no improvement after which we terminate the training loop
    hp.checkpoint_storage_location,    #location to store checkpoint
)

#Print out hte smallest validation loss
print(f"Smallest Val loss is: {smallest_val_loss}")

# #Save graphs of the losses
graph_losses(train_losses, "Train", hp.graph_save_location)
graph_losses(val_losses, "Validation", hp.graph_save_location)