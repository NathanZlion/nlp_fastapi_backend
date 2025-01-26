from etnltk import Amharic
from etnltk.lang.am import normalize
from LSTM_next_word import LSTMNextWord
from fastapi import HTTPException

class LSTMService:

    def __init__(self):
        
        self.next_word = LSTMNextWord()

    def next_words(self, text):
        doc = Amharic(text)
        words = doc.words

        if len(words) <= 2:
            raise HTTPException(status_code=400, detail="The context is too short(less than 3 words)")

        words = [normalize(word) for word in words]
        context = words[-3:]

        next_word = self.next_word.next_word_suggestion(context)
        
        if next_word:
            return next(iter(next_word))

        return ""

    def next_words_top_counts(self, text, k):

        doc = Amharic(text)
        words = doc.words

        if len(words) <= 2:
            raise HTTPException(status_code=400, detail="The context is too short(less than 3 words)")

        words = [normalize(word) for word in words]
        context = words[-3:]

        

        next_words = self.next_word.next_word_suggestion(context)

        possible_next_words = [word for word, _ in next_words.items()][:k]
        return possible_next_words



                

                


        



