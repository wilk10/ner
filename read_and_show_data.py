import json
from utils.data import Data
from colorama import Fore, Style


class DataToShow:
    def __init__(self, file_type):
        self.file_type = file_type
        assert self.file_type in ['training', 'validation', 'predictions']
        self.data = Data()
        self.predictions_path = self.data.dict_dir / 'predictions.json'

    def load_predictions(self):
        with open(str(self.predictions_path), encoding='utf-8') as f:
            data = json.load(f)
        return data

    def read_and_show(self):
        count_by_bioconcept = {}
        for kingdom in ['plant', 'animal']:
            if self.file_type in ['training', 'validation']:
                items = self.data.read_json(kingdom, self.file_type)
            else:
                items = self.load_predictions()
            for i, item in enumerate(items['result']):
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content']
                output_text = text
                annotations = item['results']['annotations']
                sorted_annotations = sorted(annotations, key=lambda a: a['start'])
                already_marked = []
                for annotation in sorted_annotations:
                    named_entity = f"{text[annotation['start']:annotation['end']]}"
                    tag = annotation['tag'].upper()
                    if tag in count_by_bioconcept.keys():
                        count_by_bioconcept[tag] += 1
                    else:
                        count_by_bioconcept[tag] = 1
                    if named_entity.lower() not in already_marked:
                        coloured_entity = f"{Fore.RED}{named_entity} {Fore.GREEN}({tag}){Style.RESET_ALL}"
                    else:
                        coloured_entity = f"{Fore.RED}{named_entity}{Style.RESET_ALL}"
                    output_text = output_text.replace(named_entity, coloured_entity)
                    already_marked.append(named_entity.lower())
                print(f'{i}: {output_text}\n --------------------------------------')
                input()
        print(count_by_bioconcept)


if __name__ == '__main__':
    data_to_show = DataToShow('predictions')
    data_to_show.read_and_show()
