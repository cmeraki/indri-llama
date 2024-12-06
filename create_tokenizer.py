import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, List

from transformers import AutoTokenizer, AutoFeatureExtractor, MimiModel
from configs import Config as cfg
from commons import MIMI, TEXT, CACHE_DIR, SPEAKER_FILE

def replace_consecutive(arr):
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    return arr[mask]

def codebook_encoding(arr: np.ndarray, per_codebook_size: int):
    c, n = arr.shape
    i_values = np.arange(c) * per_codebook_size
    arr += i_values.reshape(c, 1)
    flat_arr = arr.reshape(c * n, order='F')
    return flat_arr

def speaker_id_format(dataset, speaker):
    speaker_id = cfg.UNKNOWN_SPEAKER_ID
    if speaker:
        speaker_id = f'[spkr_{dataset}_{speaker}]'
    return speaker_id

class MimiTokenizer:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi", device=device)
        
        self.tokenizer = AutoTokenizer.from_pretrained('Llama_tokenizer', legacy=False)
        self.sampling_rate = self.feature_extractor.sampling_rate

    @torch.inference_mode()
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            raw_audio=waveform, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        output = self.model.encode(
            inputs["input_values"], 
            inputs["padding_mask"], 
            num_quantizers=cfg.n_codebooks
        )
        return output.audio_codes[0].cpu().numpy()

    def load_acoustic_tokens(self, tokens_dir: Union[str, Path]) -> Optional[List[np.ndarray]]:
        tokens_dir = Path(tokens_dir)
        if not tokens_dir.exists() or not tokens_dir.is_dir():
            print(f"Directory not found: {tokens_dir}")
            return None

        token_files = list(tokens_dir.glob('*.npy'))
        
        if not token_files:
            print(f"No .npy files found in {tokens_dir}")
            return None

        processed_tokens = []
        for token_file in token_files:
            try:
                tokens = np.load(str(token_file)).astype(np.int64)
                tokens = tokens[:cfg.n_codebooks]
                tokens = codebook_encoding(tokens, cfg.per_codebook_size)
                tokens = np.reshape(tokens, -1)                
                tokens = tokens + cfg.OFFSET[MIMI]               
                processed_tokens.append(tokens)
            except Exception as e:
                print(f"Error processing {token_file}: {e}")
        
        return processed_tokens

def get_tokenizer(type: str, device: str = 'cpu') -> MimiTokenizer:
    if type == MIMI:
        return MimiTokenizer(device=device)
    
    if type == TEXT:
        tokenizer = MimiTokenizer(device=device)
        tokenizer.tokenizer = AutoTokenizer.from_pretrained('Llama_tokenizer', legacy=False)
        return tokenizer
    
    raise ValueError(f"Unsupported tokenizer type: {type}")

def get_text_tokenizer():
    text_tokenizer = get_tokenizer(type=TEXT, device='cpu')
    
    for idx in range(cfg.VOCAB_SIZES[MIMI]):
        text_tokenizer.tokenizer.add_tokens(f'[aco_{idx}]')
    
    for tok in list(cfg.MODALITY_TOKENS.values()) + list(cfg.TASK_TOKENS.values()) + [cfg.STOP_TOKEN]:
        text_tokenizer.tokenizer.add_tokens(tok)
    
    text_tokenizer.tokenizer.add_tokens(cfg.UNKNOWN_SPEAKER_ID)
    
    speaker_tokens_path = Path(SPEAKER_FILE)
    if not speaker_tokens_path.exists():
        print(f"Warning: {SPEAKER_FILE} does not exist. Creating an empty file.")
        speaker_tokens_path.touch()
    
    for line in open(speaker_tokens_path):
        sample = json.loads(line.strip())
        speaker_id = speaker_id_format(dataset=sample['dataset'], speaker=sample['speaker'])
        text_tokenizer.tokenizer.add_tokens(speaker_id)
    
    tokenizer_save_path = 'tts_tokenizer/'
    text_tokenizer.tokenizer.save_pretrained(tokenizer_save_path)
    print('Saved tokenizer')
    
    return text_tokenizer


if __name__ == "__main__":
    text_tokenizer = get_text_tokenizer()
    mimi_tokenizer = get_tokenizer(MIMI)
    sample_tokens_dir = Path(f'{CACHE_DIR}/lj_speech/mimi/')
    
    if sample_tokens_dir.exists():
        acoustic_tokens_list = mimi_tokenizer.load_acoustic_tokens(sample_tokens_dir)
        
        if acoustic_tokens_list:
            print(f"Loaded {len(acoustic_tokens_list)} acoustic token files")
            for i, tokens in enumerate(acoustic_tokens_list):
                print(f"Acoustic tokens {i} shape: {tokens.shape}")
    
    print(f"Final Text Vocab Size: {text_tokenizer.tokenizer.vocab_size}")