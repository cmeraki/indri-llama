import torch
import random
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Literal, Union

ModalityType = Literal['mimi', 'text', 'audio', 'annotation']
TaskType = Literal['convert', 'continue']

class ConfigManager:
    MIMI: ModalityType = 'mimi'
    TEXT: ModalityType = 'text'
    AUDIO: ModalityType = 'audio'
    ANNOTATIONS: ModalityType = 'annotation'

    SEED: int = 1337

    N_CODEBOOKS: int = 8
    PER_CODEBOOK_SIZE: int = 2048

    def __init__(self):
        self._set_random_seeds()
        self.DEVICE: str = self._determine_device()
        self.DTYPE: str = self._determine_dtype()
        self.DEVICE_TYPE: str = 'cuda' if 'cuda' in self.DEVICE else 'cpu'
        
        self.PTDTYPE: torch.dtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[self.DTYPE]

        self.CTX = (
            nullcontext() if self.DEVICE_TYPE == 'cpu' 
            else torch.autocast(device_type=self.DEVICE_TYPE, dtype=self.PTDTYPE)
        )

        self.CACHE_DIR: Path = self._setup_cache_directory()
        self.VOCAB_SIZES: Dict[ModalityType, int] = {
            self.TEXT: 128000,
            self.MIMI: self.PER_CODEBOOK_SIZE * self.N_CODEBOOKS,
        }

        self.VOCAB_OFFSET: Dict[ModalityType, int] = {
            self.TEXT: 0,
            self.MIMI: self.VOCAB_SIZES[self.TEXT],
        }

        self.TASK_TOKENS: Dict[TaskType, str] = {
            'convert': '[convert]',
            'continue': '[continue]',
        }

        self.MODALITY_TOKENS: Dict[ModalityType, str] = {
            self.TEXT: '[text]',
            self.MIMI: '[mimi]',
        }

        self.UNKNOWN_SPEAKER_ID: str = '[spkr_unk]'
        self.STOP_TOKEN: str = '[stop]'
        self.VOCAB_SIZE: int = sum(self.VOCAB_SIZES.values())
        self._log_configuration()

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _determine_device(self) -> str:
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def _determine_dtype(self) -> str:
        return (
            'bfloat16' 
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported() 
            else 'float16'
        )

    def _setup_cache_directory(self) -> Path:
        cache_dir = Path("~/.cache/indri/").expanduser()
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def _log_configuration(self) -> None:
        print(f'Device: {self.DEVICE}')
        print(f'Data Type: {self.DTYPE}')
        print(f'Cache Directory: {self.CACHE_DIR}')
        print(f'Total Vocabulary Size: {self.VOCAB_SIZE}')
        print(f'Vocabulary Offset: {self.VOCAB_OFFSET}')

    def get_token(
        self, 
        token_type: Union[ModalityType, TaskType], 
        default: str = None
    ) -> str:
        tokens = {**self.MODALITY_TOKENS, **self.TASK_TOKENS}
        return tokens.get(token_type, default)


Config = ConfigManager()

MIMI = Config.MIMI
TEXT = Config.TEXT
AUDIO = Config.AUDIO
ANNOTATIONS = Config.ANNOTATIONS
TOKENS = 'tokens'
CONVERT = 'convert'
CONTINUE = 'continue'
SPEAKER_FILE = 'allowed_speakers.jsonl'