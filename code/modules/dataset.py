#Regular Imports
import re
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


#Concatenate the instruction and the input
def concat_inst_input(my_dict):
    if len(my_dict["input"]) == 0:
        return {"user" : my_dict["instruction"]}
    
    #Put the input at the top 50% of the time and at the bottom 50% of the time.
    if torch.randint(2,size=(1,)).item() == 0:
        user = my_dict["instruction"] + "\n" + my_dict["input"]
    else:
        user = my_dict["input"] + "\n" + my_dict["instruction"]
    
    return {"user" : user}

#Add the <|user|>, <|/user|>, <|output|>, and <|/output|> special tokens
def add_tokens(my_dict):
    user = f"<|user|>{my_dict['user']}<|/user|><|output|>{my_dict['output']}<|/output|>"
    return {"tokens" : user}

def mask_input(y, output_token_id, pad_token_id):
    #This function converts all tokens ids to the left (and including) the last output_token_id to be pad_token_id, so that we only perform CELoss on the desired output, and not also the user input.
    idx = torch.arange(y.shape[1], device=y.device)
    last_idx = ((y == output_token_id) * idx).amax(dim=1)
    mask = idx.unsqueeze(0) <= last_idx.unsqueeze(1)
    y[mask] = pad_token_id
    return y

#Custom collate_fn function
def collate_wrapper(tokenizer, output_token_id):
    """
    The collate_wrapper simply exposes the tokenizer and the output_token_id to the collate_fn function.
    The output_token_id is the token that marks where the assistant starts to talk.
    This is needed because we only want to apply sft to tokens where the assistant is talking, and not to the input question.
    We thus turn all tokens in the y vector before the final output token id to the pad token id, because the pad token id gets ignored in the CELoss function.
    """
    def collate_fn(batch):
        #Tokenize the strings
        list_of_strings = [text["tokens"] for text in batch]
        x = tokenizer(list_of_strings, return_tensors="pt", truncation = True, padding = True)
        
        #Create the y vector
        y = torch.roll(x["input_ids"], shifts = -1, dims = 1)          #Shift each row of x over to the left one
        y[:,-1] = tokenizer.pad_token_id                               #Set the final column of y to be the pad token because there is nothing to predict for the final token of each sentence.
        y = mask_input(y, output_token_id, tokenizer.pad_token_id)   #We want to set all tokens that are from the input of the sequence to be the pad token id so they will be ignored in the CELoss.
        
        return x, y
    return collate_fn

#Return the data loaders
def return_loaders(dataset_name, tokenizer, output_token_id, test_size, batch_size, pin_memory, num_workers):
    dataset = load_dataset(dataset_name, split="train")
    data = dataset.train_test_split(test_size = test_size)
    data = data.map(concat_inst_input, remove_columns = ["instruction", "input"])  #It now has columnes: user, output
    data = data.map(add_tokens, remove_columns = ["user", "output"])    #It now has a single column: tokens

    # #Split into train and validation datasets. The column names of the datasets are: "tokens"
    train_dataset = data["train"]
    val_dataset = data["test"]

    # #Define the train loader and the validation loader
    train_loader = DataLoader(train_dataset, collate_fn = collate_wrapper(tokenizer, output_token_id), shuffle = True, batch_size = batch_size, pin_memory = pin_memory, num_workers = num_workers)
    val_loader = DataLoader(val_dataset, collate_fn = collate_wrapper(tokenizer, output_token_id), batch_size = batch_size, pin_memory = pin_memory, num_workers = num_workers)

    return train_loader, val_loader