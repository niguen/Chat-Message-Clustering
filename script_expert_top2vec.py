
from preprocessing.preprocessing_lda import Preprocessing
from preprocessing.stop_words_de import STOP_WORDS
from evaluation.eval import Eval
import pandas as pd
import os

from models.TOP2VEC import TOP2VEC

# from evaluation.model_output import ModelOutput


# Chatbot and input Dataset
chatbot = 'roberta'
model_name = 'TOP2VEC'
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



# Clustering Parameters

'''
Available Encoder for Top2Vec[sentence_transformers]:

* doc2vec
* universal-sentence-encoder
* universal-sentence-encoder-large
* universal-sentence-encoder-multilingual
* universal-sentence-encoder-multilingual-large
* distiluse-base-multilingual-cased
* all-MiniLM-L6-v2
* paraphrase-multilingual-MiniLM-L12-v2

'''

# embedding_models = ["paraphrase-multilingual-MiniLM-L12-v2"]
embedding_models = ["doc2vec", "distiluse-base-multilingual-cased", "all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
min_cluster_size = [2, 5, 10]
min_samples = [5, 10, 15]


df_list = []
counter = 1



for embedding in embedding_models:
    for min_cluster in min_cluster_size:
        for sample in min_samples:
            config = {}
            config['name'] = f'v{counter}'
            counter = counter + 1
            config['min_cluster_size'] = min_cluster
            config['min_samples'] = sample
            config['embeddingModel'] = embedding
            

            model = TOP2VEC()
            model_output = model.train_model(data, hyperparameters=config, top_words=10)

            # Evaluate
            output_folder = os.path.join('output', chatbot, model_name, f"{model_name}_{chatbot}_{config['name']}")
            eval = Eval(output_folder=output_folder, dataset= data, model_output= model_output, name=config['name'], parameter = config)
            eval.generate_document_table()
            eval.generate_topic_table(top_n = 5)
            kpis = eval.generate_evaluation()
            
            config.update(kpis)
            df = pd.DataFrame([config])
            df_list.append(df)

            fig_path = 'wordcloud.png'
            # model.generate_topic_wordcloud(path = fig_path, topic_num = 1)
            

df = pd.concat(df_list)

# modify before print
df.drop(['min_cluster_size', 'min_samples', 'embeddingModel'], axis = 1, inplace=True)

table_str = df.to_latex(index=False, float_format="{:.3f}".format)

with open("table.txt", "w") as text_file:
    text_file.write(table_str)