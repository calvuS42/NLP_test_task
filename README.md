# Test assignment

# Stage 1

## How to run
To reproduce the training process please follow the next steps:
 1. Create the Python environment ('3.9.17' version was used in the original solution).
 2. Install all the necessary requirements (they are listed in the 'requirements.txt' file).
 3. Download into any suitable directory the training and test data from the [Kaggle competition](https://www.kaggle.com/competitions/commonlitreadabilityprize/data).
 4. Adjust the variable __'data_folder'__ (line 23 in 'train_model.py') to contain the acquired data from the previous step.
 5. Run the 'train_model.py'.

## Results
The pre-trained model is accesible via the [following link](https://drive.google.com/file/d/18w_Y8DC-JyWgziTt9AT4rDE_acfbi4SA/view?usp=sharing). The metrics after 5 epochs of train are as follows:

    Train Loss EPOCH 5: 0.0706
    Valid Loss EPOCH 5: 0.3582 

The loss in this case is a MSELoss as the model is trained for regression task (num classes == 1).

# Stage 2

## How to run
 1. Create the Python environment ('3.9.17' version was used in the original solution).
 2. Install all the necessary requirements (they are listed in the 'requirements.txt' file).
 3. Download the archived [model](https://drive.google.com/file/d/18w_Y8DC-JyWgziTt9AT4rDE_acfbi4SA/view?usp=sharing) and extract it.
 4. In the 'request.py' specify the 'MODEL_PATH' to the extracted location of the model.
 5. In terminal write the following command: ```uvicorn request:app ``` or simply run the 'request.py'.
## How to test
To test it and use the trained model for the classification the one could use 'curl' tool. The example request that was used for testing:

    curl -X POST "http://localhost:60000/classification" -H "Content-Type: application/json" -d "{\"text\": \"It is out of my power to follow a direct course without something to serve as a compass. I will go back to the village and wait till morning.\", \"id\": \"c0f722661\"}"

 The format might vary depending on terminal you are using.

The response to that request was ```{"id":"c0f722661","score":0.6013233065605164}```.