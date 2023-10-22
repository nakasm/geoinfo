import geohash
"""
"""
from bghash import Bghash

class Relation:
    def __init__(self, nodes, ways):
        self.relations = []
        self.nodes = nodes
        self.ways = ways
        self.id = None
        self.type = None
        self.highway = None
        self.name = None
        self.members = None
        self.geohashes = None
        self.coords = None
        self.tags = None

    def add_relation(self, id, members, tags):
        geohashes = set()
        bgeohashes = set()
        coords = []
        for member in members:
            osmid, _, typ = member
            if typ == 'n' and osmid in self.nodes:
                node = self.nodes[osmid]
                geohashes.add((geohash.encode(node.lat, node.lon,precision=7)))
                bgeohashes.add((Bghash(node.lat, node.lon,37)))
                coords.append((node.lat, node.lon))
            elif typ == 'w' and osmid in self.ways:
                way = self.ways[osmid]
                way_coords = [(self.nodes[node_id].lat, self.nodes[node_id].lon) for node_id in way.nodes if node_id in self.nodes]
                coords.extend(way_coords)
                for node_id in way.nodes:
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        geohashes.add((geohash.encode(node.lat, node.lon,precision=6)))
                        bgeohashes.add((Bghash(node.lat, node.lon,32)))

        self.id = id
        self.type = tags.get('type', 'No name')
        self.highway = tags.get('highway', 'No name')
        self.name = tags.get('name', 'No name')
        self.members = members
        self.geohashes = list(geohashes)
        """
        """
        self.bgeohashes = list(bgeohashes)
        self.coords = coords
        self.tags = tags

        relation = {'id': self.id, 'type': self.type, 'highway': self.highway, 'name': self.name,
                    'members': self.members, 'geohashes': self.geohashes, 'bgeohashes': self.bgeohashes,'coords': self.coords, 'tags': self.tags}
        self.relations.append(relation)

    def get_relation_ids_by_geohash(self, lat, lon):
            target_geohash = geohash.encode(lat, lon, precision=6)
            """
            """
            target_geohash = Bghash(lat, lon, 32)
            matching_ids = []
            for relation in self.relations:
                for geohs in relation['geohashes']:
                    if geohs == target_geohash:
                        matching_ids.append(relation['id'])
            return matching_ids
    
    def get_relation_by_id(self, target_id):
        for relation in self.relations:
            if relation['id'] == target_id:
                return relation
        return None  