import requests
import urllib.parse


class CatLife:
    URL = 'http://webservice.catalogueoflife.org/col/webservice'

    def __init__(self, time_to_sleep=0.5):
        self.time_to_sleep = time_to_sleep

    def get_response(self, entity):
        params = {'name': entity, 'format': 'json'}
        parsed_params = urllib.parse.urlencode(params)
        return requests.get(self.URL, params=parsed_params, timeout=10).json()


if __name__ == '__main__':
    CatLife().get_response('calonectria kyotensis')
