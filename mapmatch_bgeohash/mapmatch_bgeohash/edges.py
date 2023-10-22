from shapely import wkt

class Edge:
    def __init__(self, edge_id, start, end, length, linestring_wkt,highway,oneway,lane,name):
        corrected_linestring_wkt = linestring_wkt.replace(" 130", ", 130")
        self.edge_id = edge_id
        self.start = start
        self.end = end
        self.length = length
        self.highway = highway
        self.linestring = wkt.loads(corrected_linestring_wkt)
        self.oneway=oneway
        self.lane=lane
        self.name=name

    @staticmethod
    def is_edge_instance(other):
        return isinstance(other, Edge)

    def get_info(self):
        return{
            "edge_id": self.edge_id,
            "start":  self.start,
            "end":  self.end,
            "length":  self.length,
            "linestring":self.linestring,
            "highway":  self.highway,
            "oneway": self.oneway,
            "lane": self.lane,
            "name": self.name

        }
    
    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.start == other.start and self.end == other.end) or (self.start == other.end and self.end == other.start)
        
    def __hash__(self):
        return hash(self.edge_id)