import os
import string
from typing import List, Union
import pandas as pd

from pandas import Timedelta, Timestamp
from octis.preprocessing.preprocessing import Preprocessing as octisPreprocessing
from octis.dataset.dataset import Dataset as octisDataset


class Dataset(octisDataset):
    """Dataset class that extende the OCTIS dataset to store the original dataset before preprocessing
     
     Args:
            corpus (list, optional): List of documents after preprocessing. Defaults to None.
            original_corpus (list, optional): Original messages, without preprocessing. Defaults to None.
            vocabulary (list, optional): List of all unique words in corpus. Defaults to None.
            labels (list, optional): Labels for the dataset. Defaults to None.
            metadata (dict, optional): Additional information about the dataset. Defaults to None.
            document_indexes (list, optional): List of original document indexes. Defaults to None.

    Examples:

    ```python
    dataset =  dataset = Dataset(corpus= corpus,
                          original_corpus=message_list,
                          vocabulary= vocabulary,
                          labels = label_list,
                          metadata= None,
                          document_indexes= None )
    ```
    """


    def __init__(self, corpus=None, original_corpus=None, vocabulary=None, labels=None, metadata=None, document_indexes=None):
        super().__init__(corpus, vocabulary, labels, metadata, document_indexes)

        self.__original_corpus = original_corpus

    def get_original_corpus(self):
        return self.__original_corpus


class Preprocessing(octisPreprocessing):
    """Class for preprocessing a dataset. Extends the Octis Preproceing class by the possibility to read in data sets directly from an Excel file and to limit the dataset to a time period or number of documents.
     
        Args:
            lowercase (bool, optional): Convert dataset to lowercase. Defaults to True.
            remove_punctuation (bool, optional): Remove puntiation from datset. Defaults to True.
            lemmatize (bool, optional): Lemmatize dataset. Defaults to True.
            language (str, optional): Language of the data. Defaults to 'german'.
            stopword_list (str, optional): Custom list of stopwords or language string to apply corresponding spacy stop words. Defaults to 'german'.
            remove_numbers (bool, optional): Remove all numbers from dataset. Defaults to True.
            min_chars (int, optional): Minimum characters in a document. Defaults to 0.
            min_words_docs (int, optional): Minimum words in document. Defaults to 0.
            remove_stopwords_spacy (bool, optional): Remove the default spacy stop words from dataset(requires the language parameter). Defaults to True.
            start_date (Timestamp, optional): Startdate of messages to be considered. Defaults to None.
            time_period (Timedelta, optional): Timeperiod from startdate to be considered in the dataset. Defaults to None.
            n_samples (int, optional): Maximum number of documnets in dataset. Defaults to None.

    Examples:

    ```python
    worker = Preprocessing( lowercase= False,
                        remove_punctuation = False,
                        lemmatize = False,
                        language = 'german',
                        stopword_list = None,
                        remove_numbers = False,
                        remove_stopwords_spacy = False)

    ```
    """

    def __init__(self, 
                 lowercase: bool = True, 
                 vocabulary: List[str] = None, 
                 max_features: int = None, 
                 min_df: float = 0, 
                 max_df: float = 1, 
                 remove_punctuation: bool = True, 
                 punctuation: str = string.punctuation, 
                 remove_numbers: bool = True, 
                 lemmatize: bool = True, 
                 stopword_list: str | List[str] = 'german', 
                 min_chars: int = 1, 
                 min_words_docs: int = 0, 
                 language: str = 'german', 
                 split: bool = True, 
                 verbose: bool = False, 
                 num_processes: int = None, 
                 save_original_indexes=True, 
                 remove_stopwords_spacy: bool = True,                 
                 start_date: Timestamp = None,
                 time_period : Timedelta = None,
                 n_samples: int = None, ):
        
        self.start_date = start_date
        self.time_period = time_period
        self.max_messages = n_samples

        self.stats = {}
        self.stats['Start Date'] = start_date
        self.stats['Time Period'] = time_period
        self.stats['Max Number Messages'] = self.max_messages
        

        super().__init__(lowercase, vocabulary, max_features, min_df, max_df, remove_punctuation, punctuation, remove_numbers, lemmatize, stopword_list, min_chars, min_words_docs, language, split, verbose, num_processes, save_original_indexes, remove_stopwords_spacy)


    def dataset_from_excel(self, path: str) -> Dataset:
        """Creates on Octis Dataset from an excel File. 

        Args:
            path (str): Path to the input excel file

        Returns:
            Dataset: Preprocessed dataset according to the class settings
        """

        self.stats['File Name'] = path
        df = pd.read_excel(path, index_col=0)
       
        message_list = df['Message'].to_list()
        label_list = None
        if 'Topic' in df.columns:
            label_list = df['Topic'].to_list()
        
        # Preprocessing
        documents = [self.octisPreprocessing.simple_preprocessing_steps(document) for document in message_list]
        if self.lowercase:
             documents = [document.lower() for document in documents]
        corpus = [document.split() for document in documents]

        # generate vocabulary based on preprocessed corpus
        vocabulary = self.octisPreprocessing.filter_words(message_list)
    
        dataset = Dataset(corpus= corpus,
                          original_corpus=message_list,
                          vocabulary= vocabulary,
                          labels = label_list,
                          metadata= None,
                          document_indexes= None )

        return dataset


    def read_data(self, file_name : str) -> None:
        """Reads logfeed-csv file and selects messages accoring to the parameter, start_date, time_period and max_messages 

        Args:
            file_name (str): Path to input file
        """

        
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



    def identify_fallback(self) -> None:
        """Extracts fallback messages from conversation-Logs file"""

        self.data['is Fallback'] = ['Können Sie Ihre Frage einem Thema zuordnen?' in reply
                                    or 'Ihre Nachricht ist sehr kurz' in reply
                                    or 'Ihre Nachricht ist sehr lang.' in reply for reply in self.data['Replies'].to_list()]

        self.stats['Fallback-Messages'] = self.data.loc[self.data['is Fallback']].shape[0]
        self.df_fallback = self.data.loc[self.data['is Fallback']]

   