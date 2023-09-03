# NLP_test_task

## How to run
To reproduce the training process please follow the next steps:
 1. Create the Python environment ('3.9.17' version was used in the original solution).
 2. Install all the necessary requirements (they are listed in the 'requirements.txt' file).
 3. Download into any suitable directory the training and test data from the [Kaggle competition](https://www.kaggle.com/competitions/commonlitreadabilityprize/data).
 4. Adjust the variable __'data_folder'__ (line 23 in 'train_model.py') to contain the acquired data from the previous step.
 5. Run the 'train_model.py'.

## Results
The state dict of the model is accesible via the [following link](https://drive.google.com/file/d/1c24V1rqhIZXm9AFp6ran7htkvSy2SMf8/view?usp=sharing). The metrics after 5 epochs of train are as follows:

    Train Loss EPOCH 5: 0.0706
    Valid Loss EPOCH 5: 0.3582 

The loss in this case is a MSELoss as the model is trained for regression task (num classes == 1).