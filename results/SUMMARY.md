# Experimental Results

## Final BLEU Scores
| Model | BLEU-1 | BLEU-4 | Status |
|-------|--------|--------|--------|
| GRU-512 (Frozen CNN) | 0.2904 | 0.0503 | ✅ Trained |
| LSTM-512 (Frozen CNN) | 0.2280 | 0.0375 | ✅ Trained |
| GRU-256 (Frozen CNN) | 0.2227 | 0.0354 | ✅ Trained |
| GRU-512 (Fine-tuned CNN) | 0.2349 | 0.0415 | ✅ Trained |

## Files
- `exp1_gru.pth`: GRU-512 model
- `exp1_lstm.pth`: LSTM-512 model  
- `exp2_gru_256.pth`: GRU-256 model
- `exp3_finetuned.pth`: Fine-tuned model
- `training.log`: Complete training log
