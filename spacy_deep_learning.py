import spacy
import pandas
from spacy_with_apis import SpacyWithAPIs
from utils.data import Data
from utils.evaluation import Evaluation
from utils.kaggle_data import KaggleData
from utils.spacy_deep_model import SpacyDeepModel


class SpacyDeepLearning:
    N_ITER_FILE_NAME = 'n_iters_by_bioconcept.json'
    BIOCONCEPTS_FOR_PARTIAL_INITIALISATION = [
        'PLANT_PEST', 'PLANT_SPECIES', 'PLANT_DISEASE_COMMNAME', 'PATHOGENIC_ORGANISMS', 'TARGET_SPECIES', 'ANMETHOD']

    def __init__(self, n_kaggle_items=0):
        self.n_kaggle_items = n_kaggle_items
        self.data = Data()
        self.models_dir = self.data.cwd / SpacyDeepModel.MODELS_DIR
        n_iters_by_bioconcept_path = self.data.dict_dir / self.N_ITER_FILE_NAME
        self.n_iters_by_bioconcept = self.data.load_json(n_iters_by_bioconcept_path)

    def clean_annotations(self, text, annotations):
        cleaned_annotations = []
        for annotation in annotations:
            entity = f"{text[annotation['start']:annotation['end']]}".lower()
            bioconcept = annotation['tag'].upper().strip()
            cleaned_annotation = {'tag': bioconcept, 'start': annotation['start'], 'end': annotation['end']}
            if entity not in self.data.excluded_bioconcepts_by_entity.keys():
                cleaned_annotations.append(cleaned_annotation)
            elif bioconcept not in self.data.excluded_bioconcepts_by_entity[entity]:
                cleaned_annotations.append(cleaned_annotation)
        return cleaned_annotations

    def format_single_annotation(self, item, bioconcept):
        text = item['example']['content']
        annotations = item['results']['annotations']
        sorted_annotations = sorted(annotations, key=lambda a: a['start'])
        bioconcept_annotations = [a for a in sorted_annotations if a['tag'].upper() == bioconcept]
        clean_annotations = self.clean_annotations(text, bioconcept_annotations)
        entities = [(a['start'], a['end'], a['tag']) for a in clean_annotations]
        new_text_annotations = (text, {'entities': entities})
        return new_text_annotations

    def format_annotations(self, data, bioconcept):
        bioconcept_data = []
        for item in data['result']:
            if 'content' not in item['example'].keys():
                continue
            new_text_annotations = self.format_single_annotation(item, bioconcept)
            bioconcept_data.append(new_text_annotations)
        return bioconcept_data

    def train_models_or_get_dirs(self):
        model_dir_by_bioconcept = dict.fromkeys(self.data.bioconcepts)
        for kingdom in ['animal', 'plant']:
            training_data = self.data.read_json(kingdom, 'training')
            for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM[kingdom]:
                n_iter = self.n_iters_by_bioconcept[bioconcept]
                if bioconcept == 'LOCATION':
                    model_dir_name = f'{str(n_iter)}_clean_{self.n_kaggle_items}'
                else:
                    model_dir_name = f'{str(n_iter)}_clean'
                model_dir = self.models_dir / bioconcept.lower() / model_dir_name
                model_dir_by_bioconcept[bioconcept] = model_dir
                if not model_dir.exists():
                    bioconcept_training_data = self.format_annotations(training_data, bioconcept)
                    if bioconcept == 'LOCATION':
                        kaggle_location_data = KaggleData(self.n_kaggle_items).prepare_spacy_annotations()
                        bioconcept_training_data.extend(kaggle_location_data)
                    print(f'\n{bioconcept}: data ready, starting with deep learning training')
                    #import time
                    #start_time = time.time()
                    bioconcept_model = SpacyDeepModel(bioconcept, bioconcept_training_data, n_iter, model_dir_name)
                    bioconcept_model.train()
                    #print("--- %s seconds ---" % (time.time() - start_time))
        return model_dir_by_bioconcept

    @staticmethod
    def make_display_df(text, annotations, bioconcept=None):
        entities = [text[a['start']:a['end']] for a in annotations]
        df = pandas.DataFrame(annotations)
        df['entity'] = entities
        df['tag'] = [t.upper().strip() for t in df['tag']]
        if bioconcept is not None:
            df = df.loc[df['tag'] == bioconcept]
        return df

    def display_predictions(self, item, prediction, bioconcept):
        text = item['example']['content']
        annotations = item['results']['annotations']
        sorted_annotations = sorted(annotations, key=lambda a: a['start'])
        true_df = self.make_display_df(text, sorted_annotations, bioconcept)
        pred_df = self.make_display_df(text, prediction['pred'], bioconcept)
        print(text)
        print(f'\nANNOTATED\n{true_df}')
        print(f'\nPREDICTED\n{pred_df}')
        input()

    def predict_validation_data(self, model_dir_by_bioconcept):
        results = []
        for kingdom in ['animal', 'plant']:
            validation_data = self.data.read_json(kingdom, 'validation')
            for i, item in enumerate(validation_data['result']):
                if 'content' not in item['example'].keys():
                    continue
                item_text = item['example']['content']
                predictions = []
                for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM[kingdom]:
                    _, input_annotations = self.format_single_annotation(item, bioconcept)
                    model_dir = model_dir_by_bioconcept[bioconcept]
                    bioconcept_nlp = spacy.load(model_dir)
                    doc = bioconcept_nlp(item_text)
                    bioconcept_predictions = []
                    for entity in doc.ents:
                        entity_predictions = SpacyWithAPIs.find_matches_and_make_annotations(
                            entity.text, item_text, bioconcept)
                        bioconcept_predictions.extend(entity_predictions)
                        if bioconcept in self.BIOCONCEPTS_FOR_PARTIAL_INITIALISATION:
                            bioconcept_predictions = SpacyWithAPIs.add_partial_initials(
                                entity.text, item_text, bioconcept, bioconcept_predictions)
                    predictions.extend(bioconcept_predictions)
                if predictions:
                    predictions = SpacyWithAPIs.clean_annotations(predictions, item_text, buffer=0)
                result = {
                    'text': item_text,
                    'true': item['results']['annotations'],
                    'pred': predictions}
                results.append(result)
                print(f'item {i} of {kingdom}s predicted')
                #self.display_predictions(item, result, bioconcept)
        return results

    def run(self):
        model_dir_by_bioconcept = self.train_models_or_get_dirs()
        predictions = self.predict_validation_data(model_dir_by_bioconcept)
        evaluation = Evaluation(predictions, verbose=False)
        evaluation.run()


if __name__ == '__main__':
    SpacyDeepLearning(n_kaggle_items=4000).run()
