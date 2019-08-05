import os
import json
import pathlib


class TrainingData:
    DATA_DIR_NAME = 'Challenge_9934213_data_v2'
    DIR_NAME_BY_FLAG = {'animal': 'DATASET_AHAW_for_Challenge', 'plant': 'DATASET_PLH_for_Challenge'}
    TEXT_FILE_CHUNK = 'trainingset_text'
    JSON_FILE_CHUNK = 'trainingset_annotations'

    def __init__(self):
        self.cwd = pathlib.Path.cwd()

    def get_target_dir(self, flag):
        assert flag in ['animal', 'plant']
        dir_name = self.DIR_NAME_BY_FLAG[flag]
        return self.cwd / self.DATA_DIR_NAME / dir_name

    @staticmethod
    def get_file_path(target_dir, file_name_chunk):
        files = os.listdir(str(target_dir))
        target_files = [file for file in files if file_name_chunk in file]
        assert len(target_files) == 1
        target_file = target_files[0]
        return target_dir / target_file

    def read_text(self, flag):
        target_dir = self.get_target_dir(flag)
        file_path = self.get_file_path(target_dir, self.TEXT_FILE_CHUNK)
        with open(str(file_path), "r") as f:
            entries = [entry.rstrip() for entry in f]
        return entries

    def read_json(self, flag):
        target_dir = self.get_target_dir(flag)
        file_path = self.get_file_path(target_dir, self.JSON_FILE_CHUNK)
        with open(str(file_path), encoding='utf-8') as f:
            data = json.load(f)
        return data

    def read_and_show(self):
        for flag in ['plant', 'animal']:
            #entries = self.read_text(flag)
            entries = self.read_json(flag)
            for i, entry in enumerate(entries['result']):
                if 'content' not in entry['example'].keys():
                    continue
                text = entry['example']['content']
                print(f'{i}: {text}')
                annotations = entry['results']['annotations']
                for annotation in annotations:
                    named_entity = text[annotation['start']:annotation['end']]
                    print(f'\n{named_entity} ({annotation["tag"]})')
                print('\n ---------------------------------------------------------------------------- \n')


if __name__ == '__main__':
    TrainingData().read_and_show()
