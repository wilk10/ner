import os
import json
import time
import pathlib
import requests
import urllib.parse
from utils.data import Data


class Eppo:
    URL = 'https://data.eppo.int/api/rest/1.0'

    def __init__(self):
        self.token = os.getenv('EPPO_TOKEN', '')
        self.data = Data()
        self.entities_by_bioconcept = self.data.learn_training_entries()
        self.cwd = pathlib.Path.cwd()
        self.output_file = self.cwd / 'taxonomy_by_bioconcept.json'

    @staticmethod
    def try_except(response, input_value):
        try:
            first_result = response[0]
        except IndexError as e:
            print(f'empty response: {response} {e} with {input_value}')
            first_result = None
        except KeyError as e:
            print(f'invalid response: {response} {e} with {input_value}')
            first_result = None
        except TypeError as e:
            print(f'other error: {response} {e} with {input_value}')
            first_result = None
        return first_result

    def get_eppo_code(self, entity):
        params = {'authtoken': self.token, 'kw': entity, 'searchfor': 3, 'searchmode': 1, 'typeorg': 0}
        parsed_params = urllib.parse.urlencode(params)
        url = self.URL + f'/tools/search'
        response = requests.get(url, params=parsed_params).json()
        result = self.try_except(response, entity)
        if result is not None:
            return result['eppocode']

    def get_level1_taxonomy(self, eppo_code):
        params = {'authtoken': self.token}
        parsed_params = urllib.parse.urlencode(params)
        url = self.URL + f'/taxon/{eppo_code}/taxonomy'
        response = requests.get(url, params=parsed_params).json()
        result = self.try_except(response, eppo_code)
        if result is not None:
            level = result['level']
            assert level == 1, f'level {level} is not 1'
            return result['prefname']

    def save_data(self, data):
        with open(str(self.output_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def save_taxonomy_data_to_json(self):
        level1_taxonomy_by_bioconcept = {bioconcept: [] for bioconcept in Data.BIOCONCEPTS_BY_FLAG['plant']}
        for bioconcept, entities in self.entities_by_bioconcept.items():
            if bioconcept in level1_taxonomy_by_bioconcept.keys():
                print(f'working on {bioconcept}')
                for i, entity in enumerate(entities):
                    if len(entity) < 3:
                        continue
                    eppo_code = self.get_eppo_code(entity)
                    time.sleep(0.5)
                    if eppo_code is None:
                        continue
                    level1_taxonomy = self.get_level1_taxonomy(eppo_code)
                    time.sleep(0.5)
                    if level1_taxonomy is None:
                        continue
                    level1_taxonomy_by_bioconcept[bioconcept].append([entity, level1_taxonomy])
                self.save_data(level1_taxonomy_by_bioconcept)

    def investigate(self):
        # load_taxonomy data
        pass


if __name__ == '__main__':
    Eppo().save_taxonomy_data_to_json()
