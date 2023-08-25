
from preprocessing.preprocessing_lda import Preprocessing
from preprocessing.stop_words_de import STOP_WORDS
from evaluation.eval import Eval
import pandas as pd
import os

from models.BERTOPIC import BERTOPIC
from sentence_transformers import SentenceTransformer

# Chatbot and input Dataset
chatbot = 'roberta'
model_name = 'BERTOPIC'
dataset = os.path.join('input', chatbot, 'expert_messages - clean.xlsx')

# Preprocessing

custom_stopwords = list(STOP_WORDS)
file = os.path.join(chatbot, dataset)
worker = Preprocessing( file,
                        remove_punctuation = False,
                        lemmatize = True,
                        language = 'german',
                        stopword_list = None,
                        remove_numbers = False,
                        remove_stopwords_spacy = False)


data = worker.dataset_from_excel(dataset)


# Encoding and clustering
messageList = [" ".join(words) for words in data.get_corpus()]

# Clustering Parameters
# embedding_models = ["all-mpnet-base-v2"]
embedding_models = ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"]


nr_topics = 'auto'
min_topic_size = [2, 5, 10]
n_neighbors = [5, 10, 15]

# results = []
df_list = []
counter = 1



for embedding in embedding_models:

    # Extract embeddings
    model = SentenceTransformer(embedding)
    embeddings = model.encode(messageList, show_progress_bar=True)

    for min_topic in min_topic_size:
        for neighbor in n_neighbors:
            config = {}
            config['name'] = f'v{counter}'
            counter = counter + 1
            config['min_topic_size'] = min_topic
            config['n_neighbors'] = neighbor
            config['language'] = 'german'
            config['embeddingModel'] = embedding
            config['embeddings'] = embeddings
            

            model = BERTOPIC()
            model_output = model.train_model(data, hyperparameters=config, top_words=10)

            # Evaluate
            output_folder = os.path.join('output', chatbot, model_name, f"{model_name}_{chatbot}_{config['name']}")
            del config['embeddings']
            eval = Eval(output_folder=output_folder, dataset= data, topics=model_output['topic_values'], model_output= model_output, name=config['name'], parameter = config)
            eval.generate_document_table()
            eval.generate_topic_table(top_n = 5)
            kpis = eval.generate_evaluation()
            
            config.update(kpis)
            df = pd.DataFrame([config])
            df_list.append(df)
            

df = pd.concat(df_list)

# modify before print
df.drop(['min_topic_size', 'n_neighbors', 'embeddingModel', 'language'], axis = 1, inplace=True)

table_str = df.to_latex(index=False, float_format="{:.3f}".format)

with open("table.txt", "w") as text_file:
    text_file.write(table_str)
