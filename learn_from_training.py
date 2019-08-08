from utils.data import Data
from colorama import init, Fore, Style

init(convert=True)


class SimpleLearning:
    def __init__(self):
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()

    def fit_to_validation(self):
        output_json = {'result': []}
        results = []
        for kingdom in ['animal', 'plant']:
            validation_data = self.data.read_json(kingdom, 'validation')
            for item in validation_data['result']:
                if 'content' not in item['example'].keys():
                    continue
                text = item['example']['content'].lower()
                output_item = {'example': item['example'], 'results': {'annotations': [], 'classifications': []}}
                for bioconcept in Data.BIOCONCEPTS_BY_KINGDOM[kingdom]:
                    for entity in self.entities_by_bioconcept[bioconcept]:
                        start = text.find(entity)
                        if start != -1:
                            end = start + len(entity)
                            annotation = {'tag': bioconcept, 'start': start, 'end': end}
                            output_item['results']['annotations'].append(annotation)
                result = {
                    'text': text,
                    'true': item['results']['annotations'],
                    'pred': output_item['results']['annotations']}
                results.append(result)
                output_json['result'].append(output_item)
        return results, output_json

    def represent(self):
        items, _ = self.fit_to_validation()
        for item in items:
            text = item['text']
            output_text = item['text']
            print(f'{text}\n')
            for annotation in item['pred']:
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
