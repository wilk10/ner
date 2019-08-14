import spacy
import pathlib
import numpy as np


class SpacyDeepModel:
    MODELS_DIR = 'models'

    def __init__(self, bioconcept, train_data, n_iter):
        self.bioconcept = bioconcept
        self.train_data = train_data
        self.n_iter = n_iter
        np.random.seed(42)
        self.nlp = spacy.blank('en')
        self.ner = self.get_ner()
        self.ner.add_label(self.bioconcept)
        self.optimizer = self.nlp.begin_training()
        self.move_names = list(self.ner.move_names)
        self.output_dir = pathlib.Path.cwd() / self.MODELS_DIR / self.bioconcept.lower() / str(self.n_iter)
        assert not self.output_dir.exists()
        self.output_dir.mkdir()
        self.model_name = '_'.join([self.bioconcept.lower(), str(self.n_iter)])

    def get_ner(self):
        if 'ner' not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe('ner')
            self.nlp.add_pipe(ner)
        else:
            ner = self.nlp.get_pipe('ner')
        return ner

    def train(self):
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
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
        print('Saved model to', self.output_dir)
