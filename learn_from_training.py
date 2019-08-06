from utils.data import Data
from colorama import init, Fore, Style

init(convert=True)


class SimpleLearning:
    def __init__(self):
        self.data = Data()
        self.bioconcepts = [bc for flag, bioconcepts in Data.BIOCONCEPTS_BY_FLAG.items() for bc in bioconcepts]

    def learn_training_entries(self):
        entities_by_bioconcept = {bioconcept: [] for bioconcept in self.bioconcepts}
        for flag in ['animal', 'plant']:
            training_data = self.data.read_json(flag, 'training')
            for item in training_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content']
                annotations = item['results']['annotations']
                for annotation in annotations:
                    named_entity = f"{text[annotation['start']:annotation['end']]}"
                    clean_named_entity = named_entity.lower().strip()
                    bioconcept = annotation['tag'].upper().strip()
                    if clean_named_entity not in entities_by_bioconcept[bioconcept]:
                        entities_by_bioconcept[bioconcept].append(clean_named_entity)
        return entities_by_bioconcept

    def fit_to_validation(self):
        entities_by_bioconcept = self.learn_training_entries()
        output_json = {'result': []}
        comparisons = []
        for flag in ['animal', 'plant']:
            validation_data = self.data.read_json(flag, 'validation')
            for item in validation_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content'].lower()
                output_item = {'example': item['example'], 'results': {'annotations': [], 'classifications': []}}
                for bioconcept in Data.BIOCONCEPTS_BY_FLAG[flag]:
                    for entity in entities_by_bioconcept[bioconcept]:
                        start = text.find(entity)
                        if start != -1:
                            end = start + len(entity)
                            annotation = {'tag': bioconcept, 'start': start, 'end': end}
                            output_item['results']['annotations'].append(annotation)
                comparison = {
                    'text': text,
                    'real': item['results']['annotations'],
                    'estimated': output_item['results']['annotations']}
                comparisons.append(comparison)
                output_json['result'].append(output_item)
        return comparisons, output_json

    def represent(self):
        comparisons, _ = self.fit_to_validation()
        for item in comparisons:
            text = item['text']
            output_text = item['text']
            print(f'{text}\n')
            for annotation in item['estimated']:
                print(f'{annotation}\n')
                named_entity = f"{text[annotation['start']:annotation['end']]}"
                print(f'{named_entity}\n')
                coloured_entity = f"{Fore.RED}{named_entity}{Style.RESET_ALL}"
                output_text = output_text.replace(named_entity, coloured_entity)
                print(f'{output_text}\n --------------------------------------')
                input()

    # OK this fails because the find substring in string is too simplistic. Abort


if __name__ == '__main__':
    SimpleLearning().represent()
