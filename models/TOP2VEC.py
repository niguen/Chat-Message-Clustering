from octis.models.model import AbstractModel
from top2vec import Top2Vec

class TOP2VEC(AbstractModel):

    def __init__(self):
        super().__init__()

    def train_model(self, dataset, hyperparameters, top_words=10):



        messageList = [" ".join(words) for words in dataset.get_corpus()]

        # model
        model = Top2Vec(messageList, 
                        embedding_model= hyperparameters['embeddingModel'], 
                        hdbscan_args = {'min_cluster_size': hyperparameters['min_cluster_size'], 'min_samples': hyperparameters['min_samples']})

        
        # save and load model
        # model.save("top2vec_model")
        # model = Top2Vec.load("top2vec_model")

        topic_sizes, topic_nums = model.get_topic_sizes() 
        topics, topic_score, topic_words, word_scores = model.get_documents_topics(doc_ids =  list(range(len(messageList))), reduced=False, num_topics=1)
        topic_words, word_scores, topic_nums = model.get_topics(1)

        # delete
        topic_words1 = []
        for words in topic_words:
            topic_words1.append(['bli', 'bla', 'blub', 'bli', 'bla', 'blub', 'bli', 'bla', 'blub', 'test', 'hallo'])

        # model_output
        model_output = {"topics": topic_words1}
        model_output['topic_values'] = topics
        model_output['probabilities'] = [-1] * len(messageList)

        return model_output