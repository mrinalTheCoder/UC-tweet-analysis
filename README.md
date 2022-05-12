# Customer Review Tweets- Sentiment Analysis

## Instructions:
- Clone the repository, and run `setup.sh`
- Download the `fine-tuned` folder from [Google Drive](https://drive.google.com/drive/folders/1vrExQD1agn4wjpkA0KsdNmiAf1dz3zNO?usp=sharing)
- Place the folder in the root directory (UC-tweet-analysis)
- Run the command `$source .env/bin/activate` to enter the virtual environment (venv)
- Create a file `BEARER_TOKENS` containing the twitter API bearer token
- Run `main.py` to get a list of recent positive customer feedback using the trained model

## Training the model:
- Switch to the `tweet-dataset` branch to label a `dataset.json` file
  - Follow instructions in that branch
- Use the labelled data to finetune the model on [this Google Colab Notebook](https://colab.research.google.com/drive/1IelmuVWJepjTeqKm1ol7knojs0Vhr9Po?usp=sharing)

While in the venv, run `$deactivate` to exit
