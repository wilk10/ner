import argparse
from spacy_deep_learning import SpacyDeepLearning
from utils.data import Data


class EfsaPredictor:
    KINGDOMS = ['animal', 'plant']
    MODELS_DIR_NAME = 'final_models'
    RESULTS_DIR_NAME = 'results'
    OUTPUT_FILE_NAME_BY_KINGDOM = {'animal': 'ahaw_testset_text.json', 'plant': 'plh_testset_text.json'}

    def __init__(self, source: str):
        self.source = source
        assert self.source in ['text', 'json']
        self.data = Data()
        self.models_dir = self.data.cwd / self.MODELS_DIR_NAME
        self.results_dir = self.data.cwd / self.RESULTS_DIR_NAME
        self.test_data_by_kingdom = self.get_data()

    def get_data(self):
        test_data_by_kingdom = dict.fromkeys(self.KINGDOMS)
        for kingdom in self.KINGDOMS:
            if self.source == 'text':
                df = self.data.read_test_txt_into_df(kingdom)
                data = {'result': []}
                for i, row in df.iterrows():
                    item = {'example': {'metadata': {}}, 'results': {'annotations': [], 'classifications': []}}
                    if kingdom == 'animal':
                        metadata = {'Abstract': row.Abstract, 'Refid': row.Refid, 'Author': '', 'Title': row.Title}
                    else:
                        if len(row.content.split(' | ')) > 1:
                            text = ''.join(row.content.split(' | ')[1:])
                        else:
                            text = row.content
                        metadata = {'Refid': row.Refid, 'Author': row.Author, 'text': text, 'Title': row.Title}
                    item['example']['metadata'] = metadata
                    item['example']['content'] = row.content
                    data['result'].append(item)
            else:
                assert self.source == 'json'
                kingdom_data_path = self.data.cwd / f'{kingdom}_data.json'
                data = self.data.load_json(kingdom_data_path)
            test_data_by_kingdom[kingdom] = data
        return test_data_by_kingdom

    @staticmethod
    def format_prediction(prediction):
        if prediction['tag'] == 'ANMETHOD':
            final_tag = 'AnMethod'
        else:
            lower_tag = prediction['tag'].lower()
            final_tag = lower_tag[0].upper() + lower_tag[1:]
        return {'start': prediction['start'], 'end': prediction['end'], 'tag': final_tag}

    def run(self):
        model_dir_by_bioconcept = {bc: self.models_dir / bc.lower() for bc in self.data.bioconcepts}
        for kingdom in self.KINGDOMS:
            kingdom_data = self.test_data_by_kingdom[kingdom]
            for i, item in enumerate(kingdom_data['result']):
                item_text = item['example']['content']
                predictions = SpacyDeepLearning.make_predictions(kingdom, model_dir_by_bioconcept, item_text)
                print(f'item {i} of {kingdom}s predicted')
                formatted_predictions = [self.format_prediction(prediction) for prediction in predictions]
                kingdom_data['result'][i]['results']['annotations'] = formatted_predictions
            file_path = self.results_dir / self.OUTPUT_FILE_NAME_BY_KINGDOM[kingdom]
            self.data.save_json(file_path, kingdom_data)
            print(f'final json file save to {file_path}')


def parse_arguments():
    help_msg1 = 'if text, will predict testset txt files provided for the challenge'
    help_msg2 = 'if json, will predict custom "animal_data.json" and "plant_data.json" files, placed in ner directory'
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, choices=['text', 'json'], help=f'{help_msg1} || {help_msg2}')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    EfsaPredictor(**vars(arguments)).run()
