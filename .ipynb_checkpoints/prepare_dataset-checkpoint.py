import torch
import numpy as np
from transformers import AutoTokenizer
import json
import os
import pickle

MIMI = '[mimi]'
CONVERT = '[convert]'
CONTINUE = '[continue]'
DEFAULT_SPEAKER = '[spkr_unk]'
COMMON_STOP = '[stop]'
CACHE_DIR = '/home/meraki/.cache/indri'

# TTS Tokenizer
class TTSTokenizer:
    def __init__(self, tokenizer_name='tts_tokenizer'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)

    def encode(self, input_data, add_special_tokens=True):
        if isinstance(input_data, str):
            encoded_tokens = self.tokenizer.encode(
                input_data, 
                return_tensors='pt', 
                add_special_tokens=add_special_tokens
            )
            return encoded_tokens
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            encoded_tokens = self.tokenizer.encode(
                input_data, 
                return_tensors='pt', 
                add_special_tokens=add_special_tokens
            )
            return encoded_tokens
        else:
            raise TypeError("Input must be a string or a list of strings")

    def decode(self, tokens):
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("Input must be a torch tensor of tokens")
        
        try:
            decoded_text = self.tokenizer.decode(tokens)
            return decoded_text
        except:
            try:
                decoded_tokens = self.tokenizer.decode(tokens)
                return torch.tensor(decoded_tokens)
            except:
                raise ValueError("Unable to decode the provided tokens")

# Weave Tokens
def weave_tokens(arr : torch.tensor, per_codebook_size=2048):
    c, n = arr.shape
    i_values = np.arange(c) * per_codebook_size
    arr += i_values.reshape(c, 1)
    flat_arr = arr.reshape(c * n, order='F')
    flat_arr = flat_arr + 128255
    return torch.tensor(flat_arr, dtype=torch.int32).clone().detach()

tokenizer = TTSTokenizer()

# Final Token Sequence
@torch.no_grad()
def append_tokens(text, audio_tokens, speaker=DEFAULT_SPEAKER):
    audio_tokens = torch.tensor(audio_tokens, dtype=torch.int32).clone().detach()
    text_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int32).view(-1).clone().detach()
    convert_tokens = torch.tensor(tokenizer.encode(CONVERT, add_special_tokens=False), dtype=torch.int32).view(-1).clone().detach()
    continue_tokens = torch.tensor(tokenizer.encode(CONTINUE, add_special_tokens=False), dtype=torch.int32).view(-1).clone().detach()
    speaker_tokens = torch.tensor(tokenizer.encode(speaker, add_special_tokens=False), dtype=torch.int32).view(-1).clone().detach()
    mimi_tokens = torch.tensor(tokenizer.encode(MIMI, add_special_tokens=False), dtype=torch.int32).view(-1).clone().detach()
    stop_tokens = torch.tensor(tokenizer.encode(COMMON_STOP, add_special_tokens=False), dtype=torch.int32).view(-1).clone().detach()
    
    result = torch.cat([
        text_tokens,
        convert_tokens,
        #continue_tokens,
        speaker_tokens,
        mimi_tokens,
        audio_tokens,
        stop_tokens
    ])
    
    return result

#Load the Dataset
def load_tokens(dataset_dir):
    metadata_path = f"{CACHE_DIR}/{dataset_dir}/annotation/metadata.jsonl"
    tokens_dir = os.path.join(CACHE_DIR, dataset_dir, 'tokens', 'mimi')
    with open(metadata_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            data = json.loads(line.strip())            
            file_path = os.path.join(tokens_dir, data['id'] + '.npy')
            
            audio_tokens = np.load(file_path)
            weave_audio = weave_tokens(audio_tokens)
            yield data['raw_text'], weave_audio, data['speaker_id']

#Storing the dataset into .pkl
def store_dataset(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)    
    all_tokens = []
    for raw_text, audio_tokens, speaker in load_tokens(dataset_dir=dataset):
        with open('allowed_speakers.jsonl', 'r', encoding='utf-8') as file:
            allowed_speakers = [json.loads(line.strip()) for line in file]
        entry = next((item for item in allowed_speakers if item['dataset'] == dataset and item['speaker'] == speaker), None)
        if entry:
            combined = entry['combined']
        else:
            combined = DEFAULT_SPEAKER
        result = append_tokens(raw_text, audio_tokens, speaker=combined)
        all_tokens.append(result)
    output_file = os.path.join(output_dir, f'{dataset}_tokens.pkl')    
    with open(output_file, 'wb') as f:
        pickle.dump(all_tokens, f)
    print(f"Tokens saved to {output_file}")
    return output_file

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Name of the dataset to load')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output tokens')
    args = parser.parse_args()
    store_dataset(args.dataset, args.output_dir)
