import re
import matplotlib.pyplot as plt
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from gensim.models import Phrases
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

STOPW = {"unk", "<unk>"}
wnl = WordNetLemmatizer()


def extract_pos(doc, tag = ["NN"]):
    text = word_tokenize(doc)
    return " ".join([t[0] for t in pos_tag(text) if t[1] in tag])


def extract_bigrams(docs, biagram_model=None):
    """Extract bigrams features before POS remove
    to keep interesting patterns
    """
    list_tokens = [lemmatize(remove_stopw(doc.lower())).split() for doc in docs]
    if not biagram_model:
        biagram_model = Phrases(list_tokens, min_count=12, max_vocab_size=50000, threshold=3)
    return (biagram_model, [[b for b in biagram_model[tks] if "_" in b] for tks in list_tokens])


def lemmatize(doc):
    return " ".join([wnl.lemmatize(token) for token in doc.split()])


def remove_non_word(string):
    pattern = "^[\\w-]+$"
    return " ".join([token for token in string.split() if re.match(pattern, token)])


def remove_stopw(string):
    string = remove_stopwords(string)
    return " ".join([token for token in string.split() if token not in STOPW])


def read_text_file(fn="test.txt"):
    with open(f"data/{fn}") as f:
        return f.read()


def read_wiki(dataset="wiki.test.tokens"):
    """ In this dataset, document are separated by their title
        having format = Title here =. We split the dataset set file
        into chunks of text by the titles.
    """
    
    article = []
    begin_pattern = "^\\s*= [^=]"  # Ex. = Robert <unk> = 
    with open(f"data/{dataset}") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if re.match(begin_pattern, line):
                if len(article) > 0:
                    out = " ".join(article)
                    article = []
                    yield out
                else:
                    article.append(line.strip())
            else:
                article.append(line.strip())
        yield " ".join(article)


def plot_document_dist(lda_model, corpus, num_topics=20):
    """ Show a distribution of topics in a document
        each topic name is replaced by its top words

    """
    topic_vector = lda_model[corpus[0]]     
    topic_vector = sorted(topic_vector, key = lambda x: x[1], reverse=True)
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
    def top_words(topic):
        return "\n".join(map(lambda x: x[0], topic[:4]))
    topic_top_words = {topic[0] : f"Topic-{topic[0]}\n" + top_words(topic[1]) + "\n..." for topic in topics}

    # plot
    topic_vector = topic_vector[:5]
    x_values = [topic_top_words[t[0]] for t in topic_vector]
    x_idx = range(len(x_values))
    y_values = [t[1] for t in topic_vector]
    plt.bar(x_idx, y_values)
    plt.xticks(x_idx, x_values)
    plt.title("Distribution of topic in a document")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.xlabel("Topics")
    plt.show()
