
from preprocessing.preprocessing import Preprocessing
from preprocessing.stop_words_de import STOP_WORDS
from evaluation.eval import Eval
import pandas as pd
import os
from octis.models.NMF import NMF

# Chatbot and input Dataset
chatbot = 'roberta'
model_name = 'NMF'
dataset = os.path.join('input', chatbot, 'expert_messages - clean.xlsx')

# Preprocessing

custom_stopwords = list(STOP_WORDS)
file = os.path.join(chatbot, dataset)
worker = Preprocessing(stopword_list=custom_stopwords,
                       lowercase=True,
                       remove_punctuation=True,
                       lemmatize= True,
                       remove_numbers=True,
                       min_chars = 3,
                       min_words_docs = 2)
data = worker.dataset_from_excel(dataset)

# parameter
start = 20
end = 40
step = 2

df_list = []
parameter = {}


for x in range(start, end, step):
    print(f'value: {x}')
    
    # Model
    model = NMF(num_topics = x, random_state= 42)
    model.partitioning(use_partitions=False)
    model_output = model.train_model(dataset=data)
    topics = model_output['topic-document-matrix'].argmax(axis = 0)
    model_output['topic_values'] = topics

    # Evaluation
    output_folder = os.path.join('output', chatbot, model_name, f'{model_name}_{x}_{chatbot}')
    eval = Eval(output_folder=output_folder, dataset= data, model_output= model_output, name=f'roberta_{model_name}_{x}', parameter = parameter)
    eval.generate_document_table()
    eval.generate_topic_table(top_n = 5)
    kpis = eval.generate_evaluation()
    df = pd.DataFrame([kpis])
    df_list.append(df)


df = pd.concat(df_list)
table_str = df.to_latex(index=False, float_format="{:.3f}".format)

with open("table.txt", "w") as text_file:
    text_file.write(table_str)




