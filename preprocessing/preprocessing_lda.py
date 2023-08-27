import os
import pandas as pd

from pandas import Timedelta, Timestamp
from octis.preprocessing.preprocessing import Preprocessing as octisPreprocessing
from octis.dataset.dataset import Dataset as octisDataset


class Dataset(octisDataset):

    def __init__(self, corpus=None, original_corpus=None, vocabulary=None, labels=None, metadata=None, document_indexes=None):
        super().__init__(corpus, vocabulary, labels, metadata, document_indexes)

        self.__original_corpus = original_corpus

    def get_original_corpus(self):
        return self.__original_corpus


class Preprocessing():

    def __init__(self, 
                    lowercase: bool = True,
                    remove_punctuation: bool = True,
                    lemmatize: bool = True,
                    language: str = 'german',
                    stopword_list: str = 'german',
                    remove_numbers: bool = True,
                    min_chars: int = 0,
                    min_words_docs : int = 0,
                    remove_stopwords_spacy: bool = True,
                    start_date: Timestamp = None,
                    time_period : Timedelta = None,
                    n_samples: int = None, 
                    ) -> None:
        


        self.octisPreprocessing = octisPreprocessing(
                            lowercase=lowercase,
                            vocabulary=None, 
                            max_features=None,
                            remove_punctuation= remove_punctuation, 
                            lemmatize= lemmatize, 
                            language= language,
                            stopword_list= stopword_list,
                            split= False,
                            remove_numbers=remove_numbers,
                            min_chars = min_chars,
                            min_words_docs = min_words_docs,
                            remove_stopwords_spacy= remove_stopwords_spacy

                            )

        self.start_date = start_date
        self.time_period = time_period
        self.max_messages = n_samples

        self.stats = {}
        self.stats['Start Date'] = start_date
        self.stats['Time Period'] = time_period
        self.stats['Max Number Messages'] = self.max_messages


    def dataset_from_excel(self, path: str) -> Dataset:
        """_summary_ Creates on Octis Dataset from an excel File. 

        Args:
            path (str): _description_

        Returns:
            Dataset: Dataset
        """

        self.stats['File Name'] = path
        df = pd.read_excel(path, index_col=0)
       
        message_list = df['Message'].to_list()
        label_list = None
        if 'Topic' in df.columns:
            label_list = df['Topic'].to_list()
        
        # Preprocessing
        corpus = [self.octisPreprocessing.simple_preprocessing_steps(document).split() for document in message_list]

        # generate vocabulary based on preprocessed corpus
        vocabulary = self.octisPreprocessing.filter_words(message_list)
    
        dataset = Dataset(corpus= corpus,
                          original_corpus=message_list,
                          vocabulary= vocabulary,
                          labels = label_list,
                          metadata= None,
                          document_indexes= None )

        return dataset


    def read_data(self):
        
        csv_path = os.path.join('input', self.file_name)
        df = pd.read_csv(csv_path, delimiter=';')

        df['Received At'] = pd.to_datetime(df['Received At'])
        
        if self.max_messages is not None:
            self.data = df.loc[(df['Received At'] > self.start_date)].head(self.max_messages)
        
        elif self.time_period is not None:
            self.data = df.loc[((df['Received At'] > self.start_date) & (df['Received At'] < (self.start_date + self.time_period)))]
        
        else:
            self.data = df.loc[(df['Received At'] > self.start_date)]
        
        self.stats['Total Messages'] = self.data.shape[0]



    def identify_fallback(self):
        self.data['is Fallback'] = ['Können Sie Ihre Frage einem Thema zuordnen?' in reply
                                    or 'Ihre Nachricht ist sehr kurz' in reply
                                    or 'Ihre Nachricht ist sehr lang.' in reply for reply in self.data['Replies'].to_list()]

        self.stats['Fallback-Messages'] = self.data.loc[self.data['is Fallback']].shape[0]
        self.df_fallback = self.data.loc[self.data['is Fallback']]

   