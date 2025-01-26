from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service import LSTMService
from fastapi.middleware.cors import CORSMiddleware

lstm_service = LSTMService()


class NextWordSuggestionRequest(BaseModel):
    text: str = ""

class NextWordSuggestion(BaseModel):
    next_word: str = ""

class NextWordSuggestionTopCounts(BaseModel):
    possible_next_words: list[str] = []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/next_word")

def get_next_word(request: NextWordSuggestionRequest) -> NextWordSuggestion:
        
        text = request.text

        if text == '':
             raise HTTPException(status_code=400, detail="text should't be empty")

        nxt_word = lstm_service.next_words(text)

        if not nxt_word:

            raise HTTPException(status_code=400, detail="The context is not in the vocabulary")
        
        return NextWordSuggestion(next_word=nxt_word)

    
@app.post("/next_word/{k}")
def get_next_word_top_counts(request: NextWordSuggestionRequest, k: int) -> NextWordSuggestionTopCounts:
        
        text = request.text

        poss_nxt_word = lstm_service.next_words_top_counts(text, k)

        if not poss_nxt_word:
                
            raise HTTPException(status_code=400, detail="The context is not in the vocabulary")
        
        return NextWordSuggestionTopCounts(possible_next_words=poss_nxt_word)
