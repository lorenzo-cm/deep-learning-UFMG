import sys
import yaml
from gensim.models import Word2Vec

from config.tqdm_config import TQDMProgress
from dataset import build_corpus

EXPERIMENT_NAME = sys.argv[1]

config_yaml_path = f'config/config.yaml'

with open(config_yaml_path, 'r') as file:
    all_config = yaml.safe_load(file)
    config = all_config[EXPERIMENT_NAME]
    print("Configuration:", config)

corpus = build_corpus(config["corpus_size"])

model = Word2Vec(
    sentences=corpus,
    vector_size=config['embedding_size'],
    window=config["context_window"],
    min_count=config["min_count"],
    epochs=config['epochs'],
    workers=4,
    compute_loss=True,
    callbacks=[TQDMProgress(config['epochs'])],
)

model.save(f"models/{EXPERIMENT_NAME}.model")
