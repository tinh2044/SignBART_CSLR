import yaml

from torchviz import make_dot
from model import SignLanguageModel
from Tokenizer import GlossTokenizer

cfg_path = "configs/phoenix-2014t.yaml"

with open(cfg_path, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        
gloss_tokenizer = GlossTokenizer(config['gloss_tokenizer'])
