import torch
import torch.nn as nn
import heapq
from collections import defaultdict

class BeamSearchNode:
    def __init__(self, hidden, sequence, log_prob, length):
        self.hidden = hidden
        self.sequence = sequence  # list of token IDs
        self.log_prob = log_prob
        self.length = length
    
    def avg_log_prob(self):
        return self.log_prob / (self.length + 1e-6)
    
    def __lt__(self, other):
        return self.avg_log_prob() < other.avg_log_prob()

def beam_search_decode(decoder, features, word2idx, idx2word, 
                       beam_width=3, max_len=25, device='cuda'):
    """
    Beam search decoding for caption generation.
    """
    # Initialize beam
    start_token = word2idx['<start>']
    end_token = word2idx['<end>']
    
    # Initialize hidden state from features
    if hasattr(decoder, 'feature_to_hidden'):
        hidden = decoder.feature_to_hidden(features).unsqueeze(0)
    else:
        # For LSTM or other architectures
        hidden = features.unsqueeze(0)
    
    # Start node
    start_node = BeamSearchNode(
        hidden=hidden,
        sequence=[start_token],
        log_prob=0.0,
        length=1
    )
    
    # Beam list
    beam = [start_node]
    finished = []
    
    for step in range(max_len):
        new_beam = []
        
        for node in beam:
            # Skip if sequence already ended
            if node.sequence[-1] == end_token:
                heapq.heappush(new_beam, node)
                continue
            
            # Prepare input
            last_token = torch.tensor([[node.sequence[-1]]], device=device)
            
            # Get next token probabilities
            with torch.no_grad():
                embeddings = decoder.embed(last_token)
                
                if isinstance(decoder.gru, nn.GRU):
                    output, new_hidden = decoder.gru(embeddings, node.hidden)
                elif isinstance(decoder.lstm, nn.LSTM):
                    output, new_hidden = decoder.lstm(embeddings, node.hidden)
                else:
                    output, new_hidden = decoder.rnn(embeddings, node.hidden)
                
                logits = decoder.linear(output.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=1)
                topk_probs, topk_tokens = torch.topk(log_probs, beam_width)
            
            # Expand beam
            for i in range(beam_width):
                token = topk_tokens[0, i].item()
                token_prob = topk_probs[0, i].item()
                
                new_sequence = node.sequence + [token]
                new_log_prob = node.log_prob + token_prob
                
                new_node = BeamSearchNode(
                    hidden=new_hidden,
                    sequence=new_sequence,
                    log_prob=new_log_prob,
                    length=node.length + 1
                )
                
                heapq.heappush(new_beam, new_node)
        
        # Keep top-k nodes
        beam = heapq.nlargest(beam_width, new_beam)
        
        # Check for finished sequences
        temp_finished = []
        for node in beam:
            if node.sequence[-1] == end_token:
                temp_finished.append(node)
        
        # Move finished sequences out of beam
        for node in temp_finished:
            beam.remove(node)
            finished.append(node)
        
        # Stop if all beams are finished
        if len(beam) == 0:
            break
    
    # Combine finished and current beams
    all_candidates = finished + beam
    all_candidates.sort(reverse=True)  # Best first
    
    # Return best sequence (without <start> token)
    best_seq = all_candidates[0].sequence[1:]  # Remove <start>
    
    # Remove <end> token and everything after it
    if end_token in best_seq:
        end_idx = best_seq.index(end_token)
        best_seq = best_seq[:end_idx]
    
    return torch.tensor(best_seq).unsqueeze(0)

def add_beam_search_to_decoder(decoder_class):
    """
    Monkey-patch beam search into decoder class.
    """
    def beam_search_predict(self, features, beam_width=3, max_len=25, device='cuda'):
        return beam_search_decode(self, features, self.word2idx, None, 
                                 beam_width, max_len, device)
    
    decoder_class.beam_search_predict = beam_search_predict
    return decoder_class

