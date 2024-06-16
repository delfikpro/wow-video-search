from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

def get_embeddings(text):
    """
    Calculate the embeddings of a given string using a pre-trained BERT model.
    
    Args:
    text (str): The input string for which embeddings need to be calculated.
    
    Returns:
    torch.Tensor: The embeddings of the input string.
    """
    # Load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained('rubert_cased_L-12_H-768_A-12_pt_v1')

    # Tokenize the input text and convert to input IDs
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Load pre-trained model
    model = BertModel.from_pretrained('rubert_cased_L-12_H-768_A-12_pt_v1')

    # Get embeddings of the input string
    with torch.no_grad():
        outputs = model(input_ids)
    
    # The last hidden states used as the embeddings
    # outputs[0] contains the hidden states of each token in the input
    embeddings = outputs[0]

    return torch.mean(embeddings[0], dim=0).tolist()

def cosine_similarity(embedding1, embedding2):
    # Verify that the embeddings have the same shape
    # Average across the rows for each column to get a 1D array of length N
    # avg_embedding1 = torch.mean(embedding1[0], dim=0)
    # avg_embedding2 = torch.mean(embedding2[0], dim=0)
    
    # Ensure the embeddings have the same shape
    assert avg_embedding1.shape == avg_embedding2.shape, "Both embeddings should have the same number of columns after averaging."
    
    # Compute the cosine similarity between the two 1D arrays
    similarity = F.cosine_similarity(avg_embedding1.unsqueeze(0), avg_embedding2.unsqueeze(0))
    
    return similarity


if __name__ == '__main__':    
    text = "Hello, how are you?"
    embeddings = get_embeddings(text)
    # print("Shape of embeddings:", embeddings.shape)
    # # Note that embeddings shape will be (batch_size, sequence_length, hidden_size)
    # print("Embeddings:", embeddings)

    while True:
        s1 = input("s1: ")
        s2 = input("s2: ")

        print("sim: ", cosine_similarity(get_embeddings(s1), get_embeddings(s2)))
