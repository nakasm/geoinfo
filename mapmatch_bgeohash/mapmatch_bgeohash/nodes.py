import geohash
"""
"""
from bghash import Bghash

class Nodes:
    def __init__(self, node_id, lat, lon):
        self.node_id = node_id
        self.lat = lat
        self.lon = lon
        self.edges = []
        self.geohash = geohash.encode(lat, lon, precision=7)
        """
        """
        self.bgeohash = Bghash(lat, lon, 32)

    def get_info(self):
        return {
            'node_id': self.node_id,
            'lat': self.lat,
            'lon': self.lon
        }