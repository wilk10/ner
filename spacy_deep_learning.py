import spacy
import argparse
import datetime
from spacy_with_apis import SpacyWithAPIs
from utils.data import Data
from utils.evaluation import Evaluation
from utils.spacy_deep_model import SpacyDeepModel


class SpacyDeepLearning:
    N_ITER = 30

    def __init__(self, train_or_load, input_timestamp):
        self.train_or_load = train_or_load
        self.input_timestamp = input_timestamp
        assert self.train_or_load in ['train', 'load']
        self.data = Data()
        self.bioconcepts = self.data.bioconcepts
        self.timestamp = self.get_timestamp()
        print(f'timestamp: {self.timestamp}')
        self.models_dir = self.data.cwd / SpacyDeepModel.MODELS_DIR

    def get_timestamp(self):
        if self.train_or_load == 'load':
            assert self.input_timestamp is not None
            return self.input_timestamp
        else:
            return datetime.datetime.now().strftime('%Y_%m_%d_h%Hm%M')

    @staticmethod
    def format_single_annotation(item, bioconcept):
        text = item['example']['content']
        annotations = item['results']['annotations']
        sorted_annotations = sorted(annotations, key=lambda a: a['start'])
        bioconcept_annotations = [a for a in sorted_annotations if a['tag'].upper() == bioconcept]
        entities = [(a['start'], a['end'], a['tag'].upper()) for a in bioconcept_annotations]
        new_text_annotations = (text, {'entities': entities})
        return new_text_annotations

    @classmethod
    def format_annotations(cls, data, bioconcept):
        bioconcept_data = []
        for item in data['result']:
            if 'content' not in item['example'].keys():
                continue
            new_text_annotations = cls.format_single_annotation(item, bioconcept)
            bioconcept_data.append(new_text_annotations)
        return bioconcept_data

    def train_models(self):
        model_dir_by_bioconcept = dict.fromkeys(self.bioconcepts)
        for kingdom in ['animal', 'plant']:
            training_data = self.data.read_json(kingdom, 'training')
            for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM[kingdom]:
                bioconcept_training_data = self.format_annotations(training_data, bioconcept)
                print(f'\n{bioconcept}: data ready, starting with deep learning training')
                bioconcept_model = SpacyDeepModel(bioconcept, bioconcept_training_data, self.N_ITER, self.timestamp)
                bioconcept_model.train()
                model_dir_by_bioconcept[bioconcept] = bioconcept_model.output_dir
        return model_dir_by_bioconcept

    def compare_annotations(self, predicted, real, bioconcept):
        relevant_real = [a for a in real if a['tag'].upper().strip() == bioconcept]
        import pdb
        pdb.set_trace()

    def run(self):
        if self.train_or_load == 'train':
            model_dir_by_bioconcept = self.train_models()
        else:
            assert self.train_or_load == 'load'
            model_dir_by_bioconcept = {bc: self.models_dir / bc.lower() / self.timestamp for bc in self.bioconcepts}
        results = []
        for kingdom in ['animal', 'plant']:
            validation_data = self.data.read_json(kingdom, 'validation')
            for i, item in enumerate(validation_data['result']):
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content']
                predicted_annotations = []
                for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM[kingdom]:
                    _, input_annotations = self.format_single_annotation(item, bioconcept)
                    model_dir = model_dir_by_bioconcept[bioconcept]
                    bioconcept_nlp = spacy.load(model_dir)
                    doc = bioconcept_nlp(text)
                    annotations = [
                        {'tag': ent.label_, 'start': ent.start, 'end': ent.start + len(ent.text)}
                        for ent in doc.ents]
                    real_annotations = item['results']['annotations']
                    predicted_annotations.extend(annotations)
                    self.compare_annotations(annotations, real_annotations, bioconcept)
                if predicted_annotations:
                    predicted_annotations = SpacyWithAPIs.remove_overlapping_annotations(predicted_annotations)
                result = {
                    'text': text,
                    'true': item['results']['annotations'],
                    'pred': predicted_annotations}
                results.append(result)
                print(f'item {i} of {kingdom}s predicted')
        evaluation = Evaluation(results, verbose=False)
        evaluation.run()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_load', type=str)
    parser.add_argument('input_timestamp', type=str, nargs='?', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    SpacyDeepLearning(**vars(arguments)).run()
