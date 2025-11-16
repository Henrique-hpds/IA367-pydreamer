import queue
from collections import deque
from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor


class SequenceReplayBuffer:
    def __init__(self, capacity: int, sequence_length: int, batch_size: int):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        
    def add(self, sequence: Dict[str, np.ndarray]):
        self.buffer.append(sequence)
    
    def add_batch(self, sequences: list):
        for seq in sequences:
            self.add(seq)
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, Tensor]:
        batch_size = batch_size or self.batch_size
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough sequences in buffer: {len(self.buffer)} < {batch_size}")
        
        # Sample random indices
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        # Gather sequences
        sampled = [self.buffer[i] for i in indices]
        
        # Stack into batch
        batch = {}
        for key in sampled[0].keys():
            # Stack along batch dimension: list of (T, ...) -> (T, B, ...)
            stacked = np.stack([seq[key] for seq in sampled], axis=1)
            batch[key] = torch.from_numpy(stacked)
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    @property
    def is_ready(self):
        return len(self.buffer) >= self.batch_size


class CollectorQueue:
    def __init__(self, maxsize: int = 0):
        self.queue = queue.Queue(maxsize=maxsize)
    
    def put(self, item, block=True, timeout=None):
        self.queue.put(item, block=block, timeout=timeout)
    
    def get(self, block=True, timeout=None):
        return self.queue.get(block=block, timeout=timeout)
    
    def get_nowait(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
    
    def empty(self):
        return self.queue.empty()
    
    def qsize(self):
        return self.queue.qsize()
    
    def drain_to_buffer(self, replay_buffer: SequenceReplayBuffer, max_items: Optional[int] = None):
        count = 0
        while not self.empty() and (max_items is None or count < max_items):
            item = self.get_nowait()
            if item is not None:
                replay_buffer.add(item)
                count += 1
        return count
