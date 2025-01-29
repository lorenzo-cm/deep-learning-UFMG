import sys
import os
import yaml
from gensim.models import Word2Vec

from src.config.tqdm_config import TQDMProgress
from src.dataset import build_corpus

EXPERIMENT_NAME = sys.argv[1]

if len(sys.argv) != 2:
    print(f"usage: python3 word2vec.py NAME_OF_CONFIG")
    print(f"You can specify the config and config name in config/config.yaml")
    sys.exit(-1)

config_yaml_path = f'src/config/config.yaml'

with open(config_yaml_path, 'r') as file:
    all_config = yaml.safe_load(file)
    config = all_config[EXPERIMENT_NAME]
    print("Configuration:", config)

corpus = build_corpus(config["corpus_size"], load=True, save=False)['corpus']

model = Word2Vec(
    sentences=corpus,
    vector_size=config['embedding_size'],
    window=config["context_window"],
    min_count=config["min_count"],
    epochs=config['epochs'],
    workers=4,
    compute_loss=True,
    callbacks=[TQDMProgress(config['epochs'])],
    sg=1
)

os.makedirs('data/gensim/models/', exist_ok=True)
model.save(f"data/gensim/models/{EXPERIMENT_NAME}.model")
