
import torch
from model.LSTM_model import LSTM
import pickle

from utils.Tokenizer import Tokenizer



class LSTMNextWord:

    def __init__(self):

        self.model_path = './data/lstm_model.pth'
        self.vocab_pth = './data/vocab.pkl'
        self.build_dataset()
    

    def build_dataset(self):

        with open(self.vocab_pth, "rb") as f:
            vocab_data = pickle.load(f)


        self.tokenizer = Tokenizer(sentences=[])
        self.tokenizer.word2idx = vocab_data["word2idx"]
        self.tokenizer.idx2word = vocab_data["idx2word"]
        self.tokenizer.vocab = vocab_data["vocab"]

        self.vocab_size = len(self.tokenizer.vocab)
        self.model = LSTM(self.vocab_size)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval() 



    def next_word_suggestion(self, context : list[str]):

        if len(context) <= 2:
            return {}

        input_sequence = [self.tokenizer.encode(" ".join(context))]

        input_tensor = torch.LongTensor(input_sequence).to(torch.device('cpu'))
        
        with torch.no_grad():
            output = self.model(input_tensor)

    
        self.tokenizer.decode([50])
        self.tokenizer.vocab
        
        top_k_values, top_k_indices = torch.topk(output, self.vocab_size, dim=1)
        top_k_indices = top_k_indices.squeeze().tolist()
        top_k_values = top_k_values.squeeze().tolist()

        predicted_words = [self.tokenizer.decode([idx])[0] for idx in top_k_indices]
        
        

        top_k_values = torch.softmax(torch.tensor(top_k_values), dim=0).tolist()


        res = { word : prob for word, prob in zip(predicted_words, top_k_values)}

        return res
