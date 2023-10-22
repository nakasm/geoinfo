from nodes import Nodes
from edges import Edge
from collections import defaultdict

class GeoGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.geohash_dict = {}
        self.bgeohash_dict = {}
        self.adjacency_list = defaultdict(list)

    def add_node(self, node_id, lat, lon):
        self.nodes[node_id] = Nodes(node_id, lat, lon)
        geohash_key = Nodes(node_id, lat, lon).geohash

        """
        """
        bgeohash_key = str(Nodes(node_id, lat, lon).bgeohash)
        if bgeohash_key not in self.bgeohash_dict:
            self.bgeohash_dict[bgeohash_key] = []
        self.bgeohash_dict[bgeohash_key].append(Nodes(node_id, lat, lon).node_id)
        #print(bgeohash_key,node_id,geohash_key,type(bgeohash_key),type(geohash_key))
        """
        """

        if geohash_key not in self.geohash_dict:
            self.geohash_dict[geohash_key] = []
        self.geohash_dict[geohash_key].append(Nodes(node_id, lat, lon).node_id)
    
    def add_edge(self, edge_id, start, end, length, linestring_wkt,highway,oneway,lane,name):
        self.edges[edge_id] = Edge(edge_id, start, end, length, linestring_wkt,highway,oneway,lane,name)
        self.adjacency_list[start].append(end)
       
    def get_adjacent_nodes(self, node_id):
        return self.adjacency_list[node_id]
    
    def get_edge_info(self, start, end):
        for edge_id, edge in self.edges.items():
            if (edge.start == start and edge.end == end) or (edge.start == end and edge.end == start):
                return edge_id 
        return None, None  
    
    def get_edge_length(self, edge_id):
        if edge_id in self.edges: return self.edges[edge_id].length
        else: return None
        
    def get_edge_type(self, edge_id):
        if edge_id in self.edges: return self.edges[edge_id].highway
        else: return None

    def get_adjacent_edges_from_nodes(self, node_id1, node_id2):
        adjacent_edges = set()
        adjacent_node1=self.get_adjacent_nodes(node_id1)
        adjacent_node2=self.get_adjacent_nodes(node_id2)

        for node_id in adjacent_node1:
            edge_id=self.get_edge_info(node_id, node_id1)
            edge = self.edges[edge_id]
            if edge.start in self.get_adjacent_nodes(node_id1) or edge.end in self.get_adjacent_nodes(node_id1):
                adjacent_edges.add(edge)
                
        for node_id in adjacent_node2:
            edge_id=self.get_edge_info(node_id, node_id2)
            edge = self.edges[edge_id]
            if edge.start in self.get_adjacent_nodes(node_id2) or edge.end in self.get_adjacent_nodes(node_id2):
                adjacent_edges.add(edge)
                
        return list(adjacent_edges)
    
    def get_edges(self, node):
        adjacent_nodes = self.get_adjacent_nodes(node)
        edges = []
        for adjacent_node in adjacent_nodes:
            edge_id=self.get_edge_info(node,adjacent_node)
            edge = self.edges[edge_id]
            if edge is not None:
                edges.append(edge)
        return edges