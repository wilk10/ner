import os
import json
import time
import pathlib
import requests
import urllib.parse
from utils.data import Data


class Eppo:
    URL = 'https://data.eppo.int/api/rest/1.0'

    def __init__(self, time_to_sleep=0.5):
        self.time_to_sleep = time_to_sleep
        self.token = os.getenv('EPPO_TOKEN', '')
        self.data = Data()
        self.cwd = pathlib.Path.cwd()
        self.taxonomy_by_bioconcept_path = self.cwd / 'taxonomy_by_bioconcept.json'

    @staticmethod
    def try_except(response, input_value):
        try:
            first_result = response[0]
        except IndexError:
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
        time.sleep(self.time_to_sleep)
        if result is not None:
            return result['eppocode']

    def get_level1_taxonomy(self, eppo_code):
        params = {'authtoken': self.token}
        parsed_params = urllib.parse.urlencode(params)
        url = self.URL + f'/taxon/{eppo_code}/taxonomy'
        response = requests.get(url, params=parsed_params).json()
        result = self.try_except(response, eppo_code)
        time.sleep(self.time_to_sleep)
        if result is not None:
            level = result['level']
            assert level == 1, f'level {level} is not 1'
            return result['prefname']

    @staticmethod
    def clean_the_entity(entity):
        chunks = entity.split(' ')
        clean_chunks = [chunk for chunk in chunks if chunk != 'the']
        return ' '.join(clean_chunks)

    @staticmethod
    def check_valid_keyword(entity):
        is_long_enough = len(entity) >= 3
        has_valid_characters = all([char not in ['(', ')', '[', ']', '{', '}', '\\', ','] for char in entity])
        has_no_digits = all([not char.isdigit() for char in entity])
        return is_long_enough and has_valid_characters and has_no_digits

    def get_eppo_code_and_taxonomy(self, entity):
        clean_entity = self.clean_the_entity(entity)
        valid_keyword = self.check_valid_keyword(clean_entity)
        if valid_keyword:
            eppo_code = self.get_eppo_code(clean_entity)
            if eppo_code is not None:
                level1_taxonomy = self.get_level1_taxonomy(eppo_code)
                if level1_taxonomy is not None:
                    return level1_taxonomy
        return None

    def save_data(self, data):
        with open(str(self.taxonomy_by_bioconcept_path), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def save_taxonomy_data_to_json(self):
        entities_by_bioconcept = self.data.learn_training_entries()
        level1_taxonomy_by_bioconcept = {bioconcept: [] for bioconcept in Data.BIOCONCEPTS_BY_FLAG['plant']}
        for bioconcept, entities in entities_by_bioconcept.items():
            if bioconcept in level1_taxonomy_by_bioconcept.keys():
                print(f'working on {bioconcept}')
                for i, entity in enumerate(entities[:5]):

                    level1_taxonomy = self.get_eppo_code_and_taxonomy(entity)
                    if level1_taxonomy is not None:
                        level1_taxonomy_by_bioconcept[bioconcept].append([entity, level1_taxonomy])
                self.save_data(level1_taxonomy_by_bioconcept)


if __name__ == '__main__':
    Eppo().save_taxonomy_data_to_json()
