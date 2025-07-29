from transformers import AutoTokenizer, AutoModel
import torch

# Carregando o modelo bsb-e5-small-v2 (baseado em intfloat/e5-small-v2)
model_name = "intfloat/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Função para gerar embedding
def gerar_embedding(texto):
    # Adiciona o prefixo obrigatório para o modelo e5
    texto_formatado = f"query: {texto}"
    
    # Tokenização
    inputs = tokenizer(texto_formatado, return_tensors="pt", truncation=True, padding=True)
    
    # Geração dos embeddings (sem gradiente)
    with torch.no_grad():
        outputs = model(**inputs)

    # Média das embeddings de cada token -> embedding final da sentença
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # Converte o tensor para uma lista de floats (pronto para banco de dados)
    return embeddings.squeeze().tolist()

# Exemplo de uso
texto = "And the son of Salmon by Rahab was Boaz and the son of Boaz by Ruth was Obed and the son of Obed was Jesse"
embedding_gerado = gerar_embedding(texto)

print(f"Embedding gerado ({len(embedding_gerado)} dimensões):")
print(embedding_gerado)
