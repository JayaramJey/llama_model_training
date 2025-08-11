# Multi label classification fine-tuning 

This project fine-tunes a pretrained multi label text classification model to determine emotion in text. The model will be trained to classify whether text expresses any of the following emotions:
- Anger
- Fear
- Joy
- Sadness
- Surprise

# Project Structure
- main.py
    - Loads the dataset from the CSV file
    - Data labels are put into one array
    - Text is tokenized
    - Three different training types available (Partial, FUll freeze, fine tune)
    - Weights are determined and applied based on training data balance
    - A custom head is applied to the model (Currently using BERT)
    - The model is trained and evaluated
- custom_head.py
    - Custom classification head that is used on the model for handling multi label classification outputs
- config.yaml
    - A file which allows you to change multiple settings in training such as learning rate, training mode, ect... whithout the need to edit the code.

# Steps for training and use
### Set up instructions
- Clone the repository
- Set up a virtual environment using the following prompt
    - `conda create -n emotion-classifier`
- cd to the src file
- Active the new environment
    - `conda activate emotion-classifier`
- Install the required packages using the following command:
   - `pip install -r requirements.txt`

- Login to wandb using command line (optional)
    - `wandb login`
- download the required datafiles by using the following command line but only run once
    - `python download.py`

### Training instructions
- Edit config.yal to choose your training made:         
    - Finetune, partial, full
- Run the script using the followibng command line to perform your selected training
    - python main.py

### Loading your model for testing
- The model is saved within the frozen_bert.pt file

```
import torch
from transformers import AutoModel
from custom_head import FrozenBertClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = AutoModel.from_pretrained("bert-base-uncased")

model = FrozenBertClassifier(base_model=base_model, num_labels=5)

checkpoint = torch.load("output/frozen_bert.pt", map_location=device)
model.base_model.load_state_dict(checkpoint['base_model_state_dict'])
model.classifier.load_state_dict(checkpoint['classifier_state_dict'])

model.to(device)
model.eval()
```

# How everything works
- Bert base uncased is used as the base model
- The custom head is used to adapt the model for multi label classification
- Training is done using BCEWithLogitsLoss
- Weights are assigned to different labels to deal with the imbalance in the training data
- The config.yaml stores all the training parameters including the different training options:
    - Partial freeze: Only keep a few layers unfrozen
    - Full freeze: Keep everything frozen and only train the classifier head
    - Fine tuning: Train every layer
