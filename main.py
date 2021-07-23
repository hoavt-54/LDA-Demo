import pathlib
import logging
from gensim import models, corpora
from gensim.models import Phrases, LdaModel
from gensim.test.utils import datapath

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import utils


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger =logging.getLogger(__name__)


VOCAB_SIZE = 5000
SAVING_DIR = "saved_models"
IDX2WORD_FILE = "saved_models/dict.data"


def preprocess(data):
    """ removing stopwords, filter part-of-speech
        tokenization ...
    """
    data = [utils.remove_non_word(doc) for doc in data]
    logger.info("number of doc: {len(data)}")
    bigrams = utils.extract_bigrams(data)
    logger.info(f"bigrams0: {bigrams[0]}")
    data = [utils.extract_pos(doc) for doc in data]
    data = [doc for doc in data if 40 < len(doc) < 60000]
    data = [utils.remove_stopw(doc.lower()) for doc in data]
    data = [utils.lemmatize(doc) for doc in data]

    list_of_tokens = [doc.split() for doc in data]
    for idx, tokens in enumerate(list_of_tokens):
        tokens.extend(bigrams[idx])

    return list_of_tokens


def build_model(num_topics=30):
    data = utils.read_wiki("wiki.train.tokens")

    # preprocessing: remove too frequent words, stopwords ...
    logger.info("Start preprocessing, this will take quite some time ...")
    list_of_tokens = preprocess(data)
    print(list_of_tokens[0])

    id2word = corpora.Dictionary(list_of_tokens)
    id2word.filter_extremes(no_below=5, no_above=0.6, keep_n=VOCAB_SIZE)
    logger.info(f"Done processing dataset len, vocab len {len(id2word.keys())}, {len(list_of_tokens)}")
    

    # convert data into df vectors
    corpus = [id2word.doc2bow(tokens) for tokens in list_of_tokens]

    for num_topics in range(10, 50, 3):
        lda_model = LdaModel(corpus, num_topics=num_topics,
                                id2word=id2word,
                                passes=20,
                                iterations=400,
                                # alpha=[0.01]*num_topics,
                                alpha="auto",
                                # eta=[0.01] * VOCAB_SIZE,
                                eta="auto")
        
        # save the model
        path = pathlib.Path(f"{SAVING_DIR}/lda_topic_{num_topics}")
        path.mkdir(parents=True, exist_ok=True)
        path = path / "lda.model"
        lda_model.save(str(path.absolute()))
        id2word.save(IDX2WORD_FILE)
        

        vis = gensimvis.prepare(topic_model=lda_model, corpus=corpus, dictionary=id2word)
        pathlib.Path("lda_vizs").mkdir(parents=True, exist_ok=True)
        pyLDAvis.save_html(vis, f'lda_vizs/lda_visualization_{num_topics}.html')
    return id2word, lda_model


def inference(id2word=None, lda_model=None, num_topics=30):
    """ infer topic dist of a document given a previously trained model

    """
    
    if not id2word:
        id2word = corpora.Dictionary.load(IDX2WORD_FILE)
    
    if not lda_model:
        path = pathlib.Path(f"{SAVING_DIR}/lda_topic_30") #  there are also other models
        path = path / "lda.model"
        lda_model = LdaModel.load(str(path))


    data = utils.read_text_file("test.txt")
    list_of_tokens = preprocess([data])

    other_corpus = [id2word.doc2bow(text) for text in list_of_tokens]
    vector = lda_model[other_corpus[0]]

    utils.plot_document_dist(lda_model, other_corpus, num_topics)


def main():
    # need this, in case we use the pretrained ones
    id2word, lda_model = None, None

    id2word, lda_model = build_model()
    inference(id2word, lda_model, 45)


if __name__ == "__main__":
    main()
