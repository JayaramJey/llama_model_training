This code is used to fine tune a pretrained multi label text classification model to determine emotion in text. The model will be trained to classify whether text expresses any of the following emotions:
- Anger
- Fear
- Joy
- Sadness
- Surprise

Project Structure
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

Setup Instructions
- Install the following packages using pip install:
    - torch
    - transformers
    - datasets
    - scikit-learn 
    - wand
    - pyyaml
    - pandas
    - numpy
- Login to wandb using command line (optional)
    - wandb login

- Run the script using command line
    - python main.py

How everything works
- Bert base uncased is used as the base model
- The custom head is used to adapt the model for multi label classification
- Training is done using BCEWithLogitsLoss
- Weights are assigned to different labels to deal with the imbalance in the training data
- Select what kind of training you want
    - Partial freeze: Only keep a few layers unfrozen
    - Full freeze: Keep everything frozen and only train the classifier head
    - Fine tuning: Train every layer
