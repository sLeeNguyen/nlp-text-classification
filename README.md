# PhoBERT text classification

This project performs fine-tuning [PhoBERT](https://github.com/VinAIResearch/PhoBERT) for text classification using [VNTC dataset](https://github.com/duyvuleo/VNTC).

**Reference**

- https://phamdinhkhanh.github.io/2020/06/04/PhoBERT_Fairseq.html
- https://github.com/suicao/PhoBert-Sentiment-Classification

## Model Training

- Google Colab: https://colab.research.google.com/drive/1NVHWHdiOdMPBRqcW2AqVO8LLLEsjk_3d?usp=sharing

- Resources: https://drive.google.com/drive/folders/1F4SktVNrWxS4sip8kZL90fafZ92XHP9d?usp=sharing

## Test model via fastAPI server

```bash
# Download data
chmod +x prepare.sh
./prepare.sh

# Create virtual environment
python3 -m venv ./venv
source ./venv/bin/activate

make setup  # install libraries
make dev    # run server in development
```

*Note*: Create `model/` directory and place your trained model and `labelEncoder.pkl` inside. You also need to update `model_path` and `label_encoder_path` when creating `TextClassifier` instance in `main.py`.

Server will be running on `http://127.0.0.1:8000`.

Open browser and navigate to `http://127.0.0.1:8000/docs` (Swagger UI). Now, we can test the model via api.

<img src='./example.png' />