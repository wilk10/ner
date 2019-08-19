import spacy
import pathlib
import numpy as np


class SpacyDeepModel:
    MODELS_DIR = 'models'
    PIPES = ['ner', 'sentencizer']

    def __init__(self, bioconcept, train_data, n_iter, model_dir_name, split_into_sentences):
        self.bioconcept = bioconcept
        self.train_data = train_data
        self.n_iter = n_iter
        self.model_dir_name = model_dir_name
        self.split_into_sentences = split_into_sentences
        np.random.seed(42)
        self.nlp = spacy.blank('en')
        self.add_pipes_and_label()
        self.optimizer = self.nlp.begin_training()
        self.output_dir = pathlib.Path.cwd() / self.MODELS_DIR / self.bioconcept.lower() / self.model_dir_name
        assert not self.output_dir.exists()
        self.output_dir.mkdir()
        self.model_name = '_'.join([self.bioconcept.lower(), self.model_dir_name])

    def add_pipes_and_label(self):
        ner = self.nlp.create_pipe('ner')
        ner.add_label(self.bioconcept)
        self.nlp.add_pipe(ner)
        if self.split_into_sentences:
            sentencizer = self.nlp.create_pipe('sentencizer')
            self.nlp.add_pipe(sentencizer)

    def train(self):
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in self.PIPES]
        with self.nlp.disable_pipes(*other_pipes):
            sizes = spacy.util.compounding(1.0, 4.0, 1.001)
            for itn in range(self.n_iter):
                np.random.shuffle(self.train_data)
                batches = spacy.util.minibatch(self.train_data, size=sizes)
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(texts, annotations, sgd=self.optimizer, drop=0.35, losses=losses)
                print(f'{itn+1}/{self.n_iter}: Losses', losses)
        self.nlp.meta['name'] = self.model_name
        self.nlp.to_disk(self.output_dir)
        print(f'Saved model to {self.output_dir}')
