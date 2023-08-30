import pandas as pd
import os
import numpy as np

import os

import sklearn.metrics
import logging

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.dataset.dataset import Dataset
import string
from collections import Counter

class Eval():
    """Evaluation class for genarating kpis and tables.
    Args:
            output_folder (string): Folder to save the evaöuation files
            dataset (Dataset): Octis dataset with messages and preprocessed data
            model_output (dict): Output of the model
            name (str): Name of the trail
            parameter (dict): Corresponding model parameters

    Examples:

    ```python
    evaluation = eval('./test/', dataset, model_output, 'v1')
    ```
    """

    def __init__(self, output_folder: string, dataset: Dataset, model_output: dict, name: str, parameter: dict) -> None:
        
        self.output_folder = output_folder
        self.dataset = dataset
        self.model_output = model_output
        self.name = name
        self.numerical_labels = None
        self.parameter = parameter
    

        # self.output_folder = os.path.join(output_folder)      
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def generate_numerical_labels(self) -> None:
        """ Generates numerical representation of expert labels. """
        

        if self.numerical_labels is None:
            # convert labels into a numerical format
            labels = self.dataset.get_labels()
            df = pd.DataFrame(labels, columns=['labels'])
            
            unique_labels = list(dict.fromkeys(labels))
            label_indices = list(range(0, len(unique_labels)))
            labelMapping = dict(zip(unique_labels, label_indices))
            self.numerical_labels = df.replace(labelMapping)['labels'].to_list()
        return self.numerical_labels

    def generate_document_table(self) -> None:
        """ Generates a table with all documents (messages) and information about the topic assigned by the model as well as the expert labels. """
        logging.info('Generate document Table')

        self.generate_numerical_labels()
        df = pd.DataFrame.from_dict({'message': self.dataset.get_original_corpus(), 'topic': self.model_output['topic_values'], 'labels': self.dataset.get_labels(),'numerical Labels' : self.generate_numerical_labels()})
        excel_path = os.path.join(self.output_folder, 'documents.xlsx')
        df.to_excel(excel_path)
        
        
    def generate_topic_table(self, top_n : int = 10) -> None:
        """Create a list of the most important words for each topic, as well as the number of words per topic.

        Args:
            top_n (int, optional): The number of most important words to be displayed in the table. Defaults to 10.
        """
        logging.info('Generate topic Table')

        topic_words = [', '.join(words[:top_n]) for words in self.model_output['topics']]
        df = pd.DataFrame(topic_words, columns= ['Topic words'])
        df.index.name = 'Topic'
        
        test_dict = Counter(self.model_output['topic_values'])
        df['Message count'] = df.index.map(test_dict)

        excel_path = os.path.join(self.output_folder, 'topics.xlsx')
        df.to_excel(excel_path)


    def rand_score(self) -> float:
        """Rand score for the evaluation of clustering. The implementation of sklearn is used for the calculation.

        Returns:
            float: Value of the rand score.
        """

        return sklearn.metrics.rand_score(self.model_output['topic_values'], self.generate_numerical_labels())

    def mutual_info_score(self) -> float:
        """Mutual information score for the evaluation of topic modeling. The implementation of sklearn is used for the calculation.

        Returns:
            float: Value of the mutual information score.
        """

        return sklearn.metrics.mutual_info_score(labels_true = self.generate_numerical_labels(), labels_pred= self.model_output['topic_values'])

    def f1_score(self) -> float:
        """F1 score for the evaluation of clustering. The implementation of sklearn is used for the calculation. 
           For the evaluation of multiclass models, the average is calculated as macro.

        Returns:
            float: Value of the F1 score.
        """
        
        return sklearn.metrics.f1_score(y_true= self.generate_numerical_labels() , y_pred= self.model_output['topic_values'], average= 'macro') 

    def generate_evaluation(self) -> pd.DataFrame:
        """Creates a csv file containing evaluation metrics and the parameters for the respective test run.

        Returns:
            pd.DataFrame: Dataframe with the respective key figures to create the overall overview.
        """
        
        kpis = {}
        kpis.update(self.parameter)
        # kpis['name'] = self.name

        kpis['Nr topics'] = len(set(self.model_output['topic_values']))

        kpis['F1 score'] = self.f1_score()
        kpis['Rand score'] = self.rand_score()

        topic_diversity = TopicDiversity(topk=10)
        
        try:
            topic_diversity_score = topic_diversity.score(self.model_output)
        except:
            topic_diversity_score = '-'
        
        kpis['Topic diversity'] = topic_diversity_score
        
        npmi = Coherence(self.dataset.get_corpus(), topk=10, measure='c_npmi')    
        try:
            npmi_score = npmi.score(self.model_output)
        except:
            npmi_score = '-'
        kpis['Coherence'] = npmi_score

        # kpis['outliers'] = modelOutput.getTopics().count(-1)
        print(kpis)
        df = pd.DataFrame(kpis, index=[0]).round(decimals=3)
        csv_path = os.path.join(self.output_folder, 'evaluation.csv')
        df.to_csv(csv_path, sep=';')
        return kpis


    def generate_vector_kpis(self, df: pd.DataFrame) -> dict:
        """Creates different metrics to evaluate the vector representation of the encoder models. Not used in the current version. (Deprecated)

        Args:
            df (pd.DataFrame): Dataframe which contains the vector representation of the individual messages in the column 'vec'.

        Returns:
            dict: Dictionary with the corresponding metrics
        """
        
        kpis = {}

        X = np.array(df['vec'].tolist())
        labels = df['topics']

        silhouette_score = sklearn.metrics.silhouette_score(X, labels=labels)
        kpis['silhouette_score'] = silhouette_score

        calinski_harabasz_score = sklearn.metrics.calinski_harabasz_score(X, labels=labels)
        kpis['calinski_harabasz_score'] = calinski_harabasz_score

        davies_bouldin_score = sklearn.metrics.davies_bouldin_score(X, labels = labels)
        kpis['davies_bouldin_score'] = davies_bouldin_score

        return kpis
    

    def generate_topic_distribution(self, statistics : dict, df: pd.DataFrame) -> dict:
        """Calculates two metrics: Inents per Topic and Topic per Intent, which represents the accuracy of a cluster with respect to predefined groups. Not used in the current version. (Deprecated)

        Args:
            statistics (dict): Dictionary containing the key 'unclassified messages'.
            df (pd.DataFrame): Dataframe containing information about the dataset and labels

        Returns:
            dict: Dictionary with the corresponding metrics
        """

        # df.rename(columns={'cluster': 'topics'}, inplace=True)
        num_topics = list(df['topics'].unique())
        columns = ['intent']
        columns.extend(num_topics)

        # result_df = pd.DataFrame(columns=columns)
        result = []
        for intent in df['intent'].unique():
            values = df.loc[df['intent'] == intent]['topics'].value_counts()
            row = {'intent': intent}
            for topic in num_topics:
                if topic in values:
                    row[topic] = values[topic]
                else:
                    row[topic] = 0
            result.append(row)
        
        topics_per_intent_df_all = pd.DataFrame.from_records(result, index = 'intent')

        topics_per_intent_df = topics_per_intent_df_all[topics_per_intent_df_all.index.isin(['303-kundendienst', '31-frage-rechnung', 'no'])]
        topics_per_intent_df.loc['Other'] = topics_per_intent_df_all[~topics_per_intent_df_all.index.isin(['303-kundendienst', '31-frage-rechnung', 'no'])].sum()
        try:
            statistics['unclassified messages'] = topics_per_intent_df[-1].sum()
            
            topics_per_intent_df.drop(columns = [-1], inplace=True)
        except:
            statistics['unclassified messages'] = 0

        
        fig = topics_per_intent_df.plot.barh(stacked=True, figsize=(17, 7))
        fig_path = os.path.join(self.output_folder, 'topics_per_intent.png')
        fig.figure.savefig(fig_path)

        intents_per_topic_df = topics_per_intent_df.transpose()

       
        fig = intents_per_topic_df.plot.barh(stacked=True, figsize=(17, 7))
        fig_path = os.path.join(self.output_folder, 'intents_per_topic.png')
        fig.figure.savefig(fig_path)

        csv_path = os.path.join(self.output_folder, 'topics_per_intent.xlsx')
        topics_per_intent_df['count'] = topics_per_intent_df.replace(0, np.nan).count(axis = 1)
        topics_per_intent_df.style.highlight_max(color = 'lightgreen', axis = 0).highlight_max(color = 'lightblue', axis = 1).to_excel(csv_path)
        topics_per_intent = topics_per_intent_df['count'].sum() / topics_per_intent_df.shape[0]
        print(f'Topics_per_intent: {topics_per_intent}')
        statistics['Topics per intent'] = topics_per_intent


        csv_path = os.path.join(self.output_folder, 'intents_per_topic.xlsx')
        intents_per_topic_df['count'] = intents_per_topic_df.replace(0, np.nan).count(axis=1)
        intents_per_topic_df.style.highlight_max(color = 'lightgreen', axis = 0).highlight_max(color = 'lightblue', axis = 1).to_excel(csv_path)
        intents_per_topic = intents_per_topic_df['count'].sum() / intents_per_topic_df.shape[0]
        print(f'Intents_per_topic: {intents_per_topic}')
        statistics['Intents per topic'] = intents_per_topic

        return statistics