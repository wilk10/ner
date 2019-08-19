import re
import pandas
import numpy as np
from utils.data import Data


class KaggleData:
    FILE_NAME = 'ner_dataset.csv'

    def __init__(self, input_n_sentences):
        self.input_n_sentences = input_n_sentences
        self.data = Data()
        self.file_path = self.data.cwd / self.FILE_NAME
        self.df = self.load_df()

    def load_df(self):
        df = pandas.read_csv(self.file_path, encoding="latin1")
        return df.fillna(method="ffill")

    @staticmethod
    def group_function(s):
        result = [
            (w, p, t) for w, p, t in zip(s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist())]
        return result

    @staticmethod
    def merge_i_geo_tags(sentences):
        np.random.shuffle(sentences)
        new_sentences = []
        for sentence in sentences:
            new_sentence = []
            skip_next = False
            check_next = False
            for i, (word, pos, tag) in enumerate(sentence[:-1]):
                next_word, next_pos, next_tag = sentence[i + 1]
                if not skip_next:
                    if tag == 'B-geo' and next_tag == 'I-geo':
                        if pos in ['NNP', 'NNPS'] or next_pos not in ['NNP', 'NNPS']:
                            new_word = " ".join([word, next_word])
                            new_sentence.append((new_word, 'NNP', 'B-geo'))
                            skip_next = True
                            check_next = True
                        else:
                            new_sentence.append((word, pos, tag))
                    elif check_next and tag == 'I-geo':
                        previous_word = new_sentence[-1][0]
                        new_word = " ".join([previous_word, word])
                        new_sentence[-1] = (new_word, 'NNP', 'B-geo')
                    else:
                        check_next = False
                        new_sentence.append((word, pos, tag))
                else:
                    skip_next = False
            new_sentence.append(sentence[-1])
            new_sentences.append(new_sentence)
        return new_sentences

    def prepare_spacy_annotations(self):
        grouped = self.df.groupby("Sentence #").apply(self.group_function)
        sentences = [sentence for sentence in grouped]
        new_sentences = self.merge_i_geo_tags(sentences)
        target_with_location = self.input_n_sentences // 3 * 2
        target_no_location = self.input_n_sentences // 3
        with_location = []
        no_location = []
        for sentence in new_sentences:
            text = " ".join([word for word, pos, tag in sentence])
            locations = [word for word, pos, tag in sentence if tag == 'B-geo' and pos == 'NNP']
            entities = []
            for location in locations:
                matches = [match for match in re.finditer(location, text)]
                annotated_matches = [(match.start(), match.end(), 'LOCATION') for match in matches]
                entities.extend(annotated_matches)
            unique_entities = list(set(entities))
            sorted_entities = sorted(unique_entities, key=lambda e: e[0])
            annotation = (text, {'entities': sorted_entities})
            if sorted_entities and len(with_location) < target_with_location:
                with_location.append(annotation)
            elif not sorted_entities and len(no_location) < target_no_location:
                no_location.append(annotation)
            elif len(with_location) >= target_with_location and len(no_location) >= target_no_location:
                break
        annotations = with_location + no_location
        np.random.shuffle(annotations)
        return annotations


if __name__ == '__main__':
    KaggleData(1200).prepare_spacy_annotations()
