from utils.data import Data
from colorama import Fore, Style


class DataToInvestigate:
    def __init__(self, bioconcept):
        self.bioconcept = bioconcept
        self.kingdom = [
            kingdom for kingdom, bioconcepts in Data.BIOCONCEPTS_BY_KINGDOM.items() if bioconcept in bioconcepts][0]
        self.data = Data()

    def investigate(self):
        entries = self.data.read_json(self.kingdom, 'training')
        results = []
        for entry in entries['result']:
            if 'content' not in entry['example'].keys():
                continue
            text = entry['example']['content']
            annotations = entry['results']['annotations']
            sorted_annotations = sorted(annotations, key=lambda a: a['start'])
            for annotation in sorted_annotations:
                named_entity = f"{text[annotation['start']:annotation['end']]}"
                tag = annotation['tag'].upper()
                if tag == self.bioconcept:
                    results.append(named_entity)
        print(f'{self.bioconcept}: {results}\n ------------- \n')

    @staticmethod
    def is_int(word):
        try:
            int(word)
        except ValueError:
            return False
        else:
            return True

    def colour_annotations(self):
        entries = self.data.read_json(self.kingdom, 'training')
        for entry in entries['result']:
            if 'content' not in entry['example'].keys():
                continue
            text = entry['example']['content']
            output_text = text
            annotations = entry['results']['annotations']
            sorted_annotations = sorted(annotations, key=lambda a: a['start'])
            for annotation in sorted_annotations:
                if not annotation['tag'].upper() == self.bioconcept:
                    continue
                named_entity = f"{text[annotation['start']:annotation['end']]}"
                coloured_entity = f"{Fore.RED}{named_entity}{Style.RESET_ALL}"
                output_text = output_text.replace(named_entity, coloured_entity)
            detected_bioconctepts = []
            words = text.split(' ')
            for word in words:
                four_digits = len(word) == 4
                is_int = self.is_int(word)
                is_valid = word[:2] == '19' or word[:2] == '20'
                if four_digits and is_int and is_valid:
                    detected_bioconctepts.append(word)
                    coloured_bioconcept = f"{Fore.GREEN}{word}{Style.RESET_ALL}"
                    output_text = output_text.replace(word, coloured_bioconcept)
            print(f'{output_text}\n --------------------------------------')


if __name__ == '__main__':
    bioconcepts = [
        'PLANT_PEST', 'PLANT_SPECIES', 'PLANT_DISEASE_COMMNAME', 'PATHOGENIC_ORGANISMS',
        'TARGET_SPECIES', 'LOCATION', 'PREVALENCE', 'YEAR', 'ANMETHOD', 'LOCATION']
    for bioconcept in bioconcepts:
        DataToInvestigate(bioconcept).investigate()
        #DataToInvestigate('YEAR').colour_annotations()
