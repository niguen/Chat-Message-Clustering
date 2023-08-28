from octis.models.model import AbstractModel
from umap import UMAP
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer



class BERTOPIC(AbstractModel):

    def __init__(self):
        super().__init__()


    def train_model(self, dataset, hyperparameters, top_words=10):

        umap_model = UMAP(n_neighbors=hyperparameters['n_neighbors'], n_components=2, 
                            min_dist=0.0, metric='cosine', random_state=42)

        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        model = BERTopic(embedding_model= hyperparameters['embeddingModel'],
            language = hyperparameters['language'], 
            min_topic_size = hyperparameters['min_topic_size'], 
            umap_model = umap_model,
            ctfidf_model = ctfidf_model)

        messageList = [" ".join(words) for words in dataset.get_corpus()]
        
        if 'embeddings' in hyperparameters:
            topics, probabilities = model.fit_transform(messageList, hyperparameters['embeddings'])

        else:
            topics, probabilities = model.fit_transform(messageList)
      
        topic_labels = model.generate_topic_labels(nr_words= top_words, separator = ', ')
        topic_labels = [topic.split(', ')[1:] for topic in topic_labels]
        # topicWords = [['x' if i == '' else i for i in topicWords] for topicWords in topic_labels] 


        hierarchical_topics = model.hierarchical_topics(messageList)
        topic_str = model.get_topic_tree(hierarchical_topics)
        with open("topic_tree.txt", "w") as text_file:
            text_file.write(topic_str)



        model_output = {"topics": topic_labels}
        model_output['topic_values'] = topics
        model_output['probabilities'] = probabilities   

        return model_output
    