# O objetivo desse é código é gerar embeddings de textos de acordo com o modelo e5-small-v2


import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Texto de entrada
input_texts = [
    "passage: amora"
]

# Carrega modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2')

# Tokeniza e gera embeddings
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# Normaliza
embeddings = F.normalize(embeddings, p=2, dim=1)

# Imprime o vetor (dimensão [1, 384])
print("Forma do embedding:", embeddings.shape)
print("Embedding (384 dimensões):")
print(embeddings[0])  # ou embeddings.squeeze(0) para remover a dimensão extra
