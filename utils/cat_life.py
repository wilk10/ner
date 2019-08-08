import requests
import urllib.parse


class CatLife:
    URL = 'http://webservice.catalogueoflife.org/col/webservice'

    def __init__(self, time_to_sleep=0.5):
        self.time_to_sleep = time_to_sleep

    def get_response(self, entity):
        params = {'name': entity, 'format': 'json'}
        parsed_params = urllib.parse.urlencode(params)
        response = requests.get(self.URL, params=parsed_params).json()
        print(response)
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    CatLife().get_response('calonectria kyotensis')
