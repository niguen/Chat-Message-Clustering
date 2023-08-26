from octis.models.model import AbstractModel
from top2vec import Top2Vec

from keybert import KeyBERT
import pandas as pd

class TOP2VEC(AbstractModel):

    def __init__(self):
        super().__init__()

        self.model = None

    def train_model(self, dataset, hyperparameters, top_words=10):



        messageList = [" ".join(words) for words in dataset.get_corpus()]

        # model
        self.model = Top2Vec(messageList, 
                        embedding_model= hyperparameters['embeddingModel'], 
                        hdbscan_args = {'min_cluster_size': hyperparameters['min_cluster_size'], 'min_samples': hyperparameters['min_samples']})

        
        # save and load model
        # model.save("top2vec_model")
        # model = Top2Vec.load("top2vec_model")

        topic_sizes, topic_nums = self.model.get_topic_sizes() 
        topics, topic_score, topic_words, word_scores = self.model.get_documents_topics(doc_ids =  list(range(len(messageList))), reduced=False, num_topics=1)
        
        # topic_words, word_scores, topic_nums = self.model.get_topics(10)
        # for words, scores, num in zip(topic_words, word_scores, topic_nums):
        #     print(num)
        #     print(f"Words: {words}")

        
        topic_mapping = pd.DataFrame({'documents': messageList, 'topic': topics})
        topic_words = []
        for topic in set(topics):
            documents = topic_mapping.loc[topic_mapping['topic'] == topic]['documents'].to_list()
            documents_str = " ".join(documents)
            words = self.get_key_words(documents_str, top_words)
            topic_words.append(words)



     
        # model_output
        model_output = {"topics": topic_words}
        model_output['topic_values'] = topics

        return model_output
    
    def generate_topic_wordcloud(self, path: str, topic_num: int):

        fig = self.model.generate_topic_wordcloud(topic_num)
        fig.savefig(path)


    def get_key_words(self, documents: str, top_n : int):

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(documents, top_n = top_n)
        return [tup[0] for tup in keywords]

