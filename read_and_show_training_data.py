from utils.data import Data
from colorama import Fore, Style


class DataToShow:
    def __init__(self):
        self.data = Data()

    def read_and_show(self):
        count_by_bioconcept = {}
        for flag in ['plant', 'animal']:
            #entries = self.read_text(flag)
            entries = self.data.read_json(flag, 'training')
            for i, entry in enumerate(entries['result']):
                if 'content' not in entry['example'].keys():
                    continue
                text = entry['example']['content']
                output_text = text
                annotations = entry['results']['annotations']
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
    DataToShow().read_and_show()
