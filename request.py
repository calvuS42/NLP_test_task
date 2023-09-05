from fastapi import FastAPI
from pydantic import BaseModel, constr, conlist
from transformers import BertTokenizerFast, BertForSequenceClassification
from typing import List
from transformers import pipeline
import torch
import uvicorn
import logging

# define the device which is used for training 
device = "cuda" if torch.cuda.is_available() else "cpu" 

MODEL_PATH = './model.pt'
# define the model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1).to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

classifier = pipeline("text-classification",
                      model=model,
                      tokenizer=tokenizer)
app = FastAPI()

class UserRequestIn(BaseModel):
    text: constr(min_length=1)
    id: constr(min_length=1)

class ScoredIDsOut(BaseModel):
    id: str
    score: float

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/classification", response_model=ScoredIDsOut)
def read_classification(user_request_in: UserRequestIn):
     # classification
    result = classifier(user_request_in.text, user_request_in.id)

    # Create the response matching the ScoredIDsOut model
    response = {
        "id": user_request_in.id,  
        "score": result[0]["score"]
    }
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=60000)