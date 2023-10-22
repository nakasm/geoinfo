import pandas as pd
from collections import defaultdict
import geohash
from geopy.distance import great_circle
from shapely import wkt
from shapely.ops import nearest_points
from shapely.geometry import LineString, Point
from typing import List
import math
import folium
from pyproj import CRS, Transformer
from shapely.ops import nearest_points
from shapely.ops import substring
import csv

epsilon = 1e-50
BETA=4.07
SIGMA_Z=3
MINIMUM=10**(-45)
MIN=10 ** (-95)



class Node:
    def __init__(self, node_id, lat, lon):
        self.node_id = node_id
        self.lat = lat
        self.lon = lon
        self.edges = []
        self.geohash = geohash.encode(lat, lon, precision=7)

    def get_info(self):
        return {
            'node_id': self.node_id,
            'lat': self.lat,
            'lon': self.lon
        }

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


class Path:
    def __init__(self, log_prob, traj,prev_path=None, edge=None, matched_position_on_prev_edge=None, matched_position_on_current_edge=None):
        self.log_prob = log_prob
        self.traj=traj
        self.prev_path = prev_path
        self.edge = edge
        self.matched_position_on_prev_edge = matched_position_on_prev_edge
        self.matched_position_on_current_edge = matched_position_on_current_edge
        self.sequence = []
        if prev_path:
            self.sequence.extend(prev_path.sequence)
        if edge:
            self.sequence.append(edge)

    def __lt__(self, other):
        return self.log_prob < other.log_prob
    
    def __eq__(self, other):
        if isinstance(other, Path):
            return self.sequence == other.sequence and (abs(self.log_prob - other.log_prob) < 1e-6)
        return False
    

    def extract_substring(self, start_point, end_point):
        start_distance = self.edge.linestring.project(Point(start_point))
        end_distance = self.edge.linestring.project(Point(end_point))
        substring_line = substring(self.edge.linestring, start_distance, end_distance, normalized=False)
        return substring_line
    
    def get_info(self):
        return {
            "prob": self.log_prob,
            "prev_path": self.prev_path,
            "traj": self.traj,
            "current_path": self.edge,
            "edge_sequence": [edge.get_info() for edge in self.sequence],
            "matched_position_on_prev_edge": self.matched_position_on_prev_edge,
            "matched_position_on_current_edge": self.matched_position_on_current_edge
        }

    

class GeoGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.geohash_dict = {}
        self.adjacency_list = defaultdict(list)

    def add_node(self, node_id, lat, lon):
        self.nodes[node_id] = Node(node_id, lat, lon)
        geohash_key = Node(node_id, lat, lon).geohash
        if geohash_key not in self.geohash_dict:
            self.geohash_dict[geohash_key] = []
        self.geohash_dict[geohash_key].append(Node(node_id, lat, lon).node_id)
    
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
    
    def find_nearest_edge(self, index):
        min_distance = float('inf')
        nearest_edge = None
        first_point = (gps_traj.iloc[index]['lat'], gps_traj.iloc[index]['lon'])
        first_point_node_ids = find_nodes_in_same_geohash(graph, gps_traj, index)
        for node_id1 in first_point_node_ids:
                node_ids=graph.get_adjacent_nodes(node_id1)
                for node_id2 in node_ids:
                    edge_id=graph.get_edge_info(node_id1,node_id2)
                    edge = graph.edges[edge_id]
                    dis=graph.calculate_shortest_distance(edge,first_point)
                    if min_distance>dis:
                        min_distance=dis
                        nearest_edge=edge
        return nearest_edge,min_distance
    
    def get_edges(self, node):
        adjacent_nodes = graph.get_adjacent_nodes(node)
        edges = []
        for adjacent_node in adjacent_nodes:
            edge_id=graph.get_edge_info(node,adjacent_node)
            edge = graph.edges[edge_id]
            if edge is not None:
                edges.append(edge)
        return edges
    

class MapMatcher:
    def __init__(self, graph, k,sigma_z=4.07, beta=1.0):
        self.graph = graph
        self.sigma_z = sigma_z
        self.beta = beta
        self.k=k

    def calculate_routedistance(self,edge1, point1, edge2, point2,intersection_node):
        transformer = Transformer.from_crs("epsg:4326", "epsg:32653")

        line1 = LineString([transformer.transform(lat, lon) for lon, lat, *_ in edge1.linestring.coords])
        line2 = LineString([transformer.transform(lat, lon) for lon, lat, *_ in edge2.linestring.coords])
        point1 = Point(transformer.transform(*point1[::-1]))
        point2 = Point(transformer.transform(*point2[::-1]))
        intersection_point = Point(transformer.transform(intersection_node.lat, intersection_node.lon))
        line1_start_to_point1 = line1.project(point1)
        line2_start_to_point2 = line2.project(point2)
        line1_start_to_node = line1.project(intersection_point)
        line2_start_to_node = line2.project(intersection_point)
        distance1 = abs(line1_start_to_node - line1_start_to_point1)
        distance2 = abs(line2_start_to_node - line2_start_to_point2)
        
        return distance1 + distance2

    def calculate_distance_on_edge(self, edge, point,t):
        edge_shape = LineString(edge.linestring)
        edge_shape = LineString([(lat, lon) for lon, lat in edge_shape.coords])
        point_shape = Point(point)
        nearest_point_on_edge = nearest_points(edge_shape, point_shape)[0]
        point1 = (nearest_point_on_edge.x, nearest_point_on_edge.y)  # y is latitude, x is longitude
        point2 = (point_shape.x, point_shape.y)
        distance = great_circle(point1, point2).meters

        return distance, (nearest_point_on_edge.y, nearest_point_on_edge.x)
    
    # A empirical distribution
    def calculate_transition_probability(self, prev_edge, prev_traj, current_edge, current_traj,t):
        _, point_on_prev_edge = self.calculate_distance_on_edge(prev_edge, prev_traj,t)
        _, point_on_current_edge = self.calculate_distance_on_edge(current_edge, current_traj,t)
        if prev_edge.end == current_edge.start:  intersection_nodeid=prev_edge.end
        if prev_edge.start == current_edge.end:  intersection_nodeid=prev_edge.start
        if prev_edge.end == current_edge.end:  intersection_nodeid=prev_edge.end
        if prev_edge.start == current_edge.start:  intersection_nodeid=prev_edge.start

        if prev_edge==current_edge: 
            dis1=great_circle((point_on_prev_edge[1],point_on_prev_edge[0]),(point_on_current_edge[1],point_on_current_edge[0])).meters
            actual_distance = great_circle(prev_traj, current_traj).meters
            delta_distance = abs(dis1 - actual_distance)
            transition_prob = (1 / BETA) * math.exp(-delta_distance)
       
            return transition_prob
        
        intersection_node=graph.nodes[intersection_nodeid]
        routedis=self.calculate_routedistance(prev_edge, point_on_prev_edge, current_edge, point_on_current_edge,intersection_node)
        actual_distance = great_circle(prev_traj, current_traj).meters
        delta_distance = abs(routedis - actual_distance)

        delta_distance = abs(routedis - actual_distance)
        transition_prob = (1 / BETA) * math.exp(-delta_distance)

        return transition_prob
    
    # A gaussian distribution
    def calculate_emission_probability(self, current_edge, current_traj,t):
        c = 1 / (SIGMA_Z * math.sqrt(2 * math.pi))
        distance_edge_traj,_=self.calculate_distance_on_edge(current_edge, current_traj,t)
        emiprob=c * math.exp(-distance_edge_traj**2)

        if current_edge.highway=="tertiary" or current_edge.highway=="motorway_link" or current_edge.highway=="['motorway_link' 'tertiary']": return 0
        return emiprob
 
    def log_prob_multiply(self, log_a, log_b):
        return log_a + log_b
    
    @staticmethod
    def paths_are_equal(path1, path2):
        if len(path1) != len(path2):
            return False

        for edge1, edge2 in zip(path1, path2):
            if edge1 != edge2:
                return False

        return True
    
    def find_initial_edge_and_point(self, graph, gps_traj, start):
        nodes_list = find_nodes_in_same_geohash(graph, gps_traj, start)

        candidate_edges_list = [edge for node in nodes_list for edge in graph.get_edges(node)]

        init_closest_point, _ = min(
            (point for edge in candidate_edges_list for point in [self.calculate_distance_on_edge(edge, gps_traj[i], i) for i in [start, start+1]]),
            key=lambda x: x[1]
        )

        return init_closest_point

    def backtrack(self, states):
        all_paths = []
        final_states = states[-1]

        for edge in final_states:
            paths = final_states[edge]
            paths.sort(key=lambda path: path.log_prob, reverse=True)
            top_k_paths = paths[:self.k]
            for path in top_k_paths:
                current_path = path
                path_edges = []
                while current_path is not None:
                    path_edges.append(current_path)  
                    current_path = current_path.prev_path

                path_edges = list(reversed(path_edges))
                all_paths.append((path_edges, path.log_prob))
            
        all_paths.sort(key=lambda x: x[1], reverse=True)
        final_k_best_paths, final_probs = zip(*all_paths[:self.k])
        
        return list(final_k_best_paths), list(final_probs)

    def match(self, trajectory, pause=False, start=0):
        states = []
        i=0
        nodes_list = find_nodes_in_same_geohash(graph, gps_traj, start)
        candidate_ini_edgeslist = [edge for node in nodes_list for edge in graph.get_edges(node)]
        maxl=-1
        for edge in candidate_ini_edgeslist:
            if edge.highway=="tertiary" or edge.highway=="motorway_link" or edge.highway=="['motorway_link' 'tertiary']": continue
            dis,point= self.calculate_distance_on_edge(edge, trajectory[start],start)
            point1=self.calculate_emission_probability(edge, trajectory[start],start)
            point2=self.calculate_emission_probability(edge, trajectory[start+4],start+4)
            point3=self.calculate_emission_probability(edge, trajectory[start+8],start+8)
            a=point1*point2*point3
            if maxl<a:
                maxl=a
                nearest_edge=edge
                init_closest_point=point
    
        #print(start,"initial_state",(nearest_edge.get_info()["start"],nearest_edge.get_info()["end"]))

        iniprob = 1
        _,init_closest_point = self.calculate_distance_on_edge(nearest_edge, trajectory[start],0)
        states.append({
            nearest_edge: [Path(iniprob, trajectory[start],None,nearest_edge,init_closest_point,init_closest_point)]
        })
        for t in range(start+1, len(trajectory)):
            prev_states = states[-1]
            current_states = defaultdict(list)  
            all_paths = []  
            if t%4!=0:continue

            if not [self.calculate_emission_probability(graph.edges[state.get_info()["edge_id"]], trajectory[t-4], t-4) for state in states[-1]]:
                return states[:-1], t

            if t>4 and max(self.calculate_emission_probability(graph.edges[state.get_info()["edge_id"]], trajectory[t-4], t-4) for state in states[-1])==0:
                    for i in states[-1]:
                            print(t,"interrupt",(i.get_info()["start"],i.get_info()["end"]))
                    if not states[:-1]:
                        return states[:-1], t+4
                    
                    return states[:-1], t-4
            for prev_edge in prev_states:
                if Edge.is_edge_instance(prev_edge)==False:   continue
                
                prev_edge_id = prev_edge.get_info()["edge_id"]
                start = self.graph.edges[prev_edge_id].start
                end = self.graph.edges[prev_edge_id].end
                candidate_edges = self.graph.get_adjacent_edges_from_nodes(start, end)
                candidate_edges = [edge for edge in candidate_edges if self.calculate_distance_on_edge(edge,trajectory[t],t)[0] < 30.0]
    
                if not candidate_edges: candidate_edges.append(prev_edge)

                for current_edge in candidate_edges:

                    if current_edge.highway=="motorway_link" or current_edge.highway=="tertiary": continue
                    for prev_path in prev_states[prev_edge]:
                        transition_prob = self.calculate_transition_probability(prev_edge, trajectory[t - 4], current_edge, trajectory[t], t)
                        emission_prob = self.calculate_emission_probability(current_edge, trajectory[t], t)
                        log_transition_prob = math.log(transition_prob + epsilon)
                        log_emission_prob = math.log(emission_prob + epsilon)
                        total_log_prob = self.log_prob_multiply(prev_path.log_prob, self.log_prob_multiply(log_transition_prob, log_emission_prob))
                        distance, prev_closest_point = self.calculate_distance_on_edge(prev_edge, trajectory[t - 4],t)
                        distance, curr_closest_point = self.calculate_distance_on_edge(current_edge, trajectory[t],t)
                        new_path = Path(total_log_prob, trajectory[t],prev_path, current_edge, prev_closest_point,curr_closest_point)

                        if new_path not in all_paths:
                            all_paths.append(new_path)
                            current_states[current_edge].append(new_path)
                            if len(current_states[current_edge]) > self.k:
                                diversity_penalty = 0.1  
                                current_states[current_edge].sort(key=lambda path: path.log_prob - diversity_penalty * min(calculate_diversity(path, p) for p in current_states[current_edge]), reverse=True)
                                current_states[current_edge] = current_states[current_edge][:self.k]
            all_paths.sort(key=lambda path: path.log_prob, reverse=True)
            top_k_paths = all_paths[:self.k]

            next_states = defaultdict(list)
            
            for path in top_k_paths:
                next_states[path.edge].append(path)
            
            states.append(next_states)

        return states,len(trajectory)


def calculate_diversity(path1, path2):
    min_length = min(len(path1.sequence), len(path2.sequence))
    matching_edges = sum(e1 == e2 for e1, e2 in zip(path1.sequence, path2.sequence))
    matching_ratio = matching_edges / min_length if min_length > 0 else 1
    diversity = 1 - matching_ratio  
    return diversity


def find_nodes_in_same_geohash(graph, gps_traj, index):
    geohash_key = gps_traj.loc[index, 'geohash']
    node_ids = []
    if geohash_key in graph.geohash_dict:
        node_ids += graph.geohash_dict[geohash_key]
    if len(node_ids) == 0:
        neighbor_geohashes = geohash.neighbors(geohash_key)
        for neighbor in neighbor_geohashes:
            if neighbor in graph.geohash_dict:
                node_ids += graph.geohash_dict[neighbor]   
    return node_ids



def draw_paths(folium_map,paths):
    for i,path_list in enumerate(paths):
        if i>0:continue
        for idx, path in enumerate(path_list):
            color = "pink"

            if path.prev_path:
                if path.prev_path.edge == path.edge:
                    edgeid=((path.prev_path.edge.get_info()["start"],path.prev_path.edge.get_info()["end"]),path.prev_path.edge.get_info()["highway"])
                    edgeid2=((path.edge.get_info()["start"],path.edge.get_info()["end"]),path.edge.get_info()["highway"])
                    substring_line = path.extract_substring(path.matched_position_on_prev_edge, path.matched_position_on_current_edge)
                    locations = [(pt[1], pt[0]) for pt in substring_line.coords]
                    folium.PolyLine(locations=locations,popup=(edgeid,edgeid2),color=color).add_to(folium_map)
                    folium.CircleMarker([path.matched_position_on_prev_edge[1], path.matched_position_on_prev_edge[0]],radius=1,popup=edgeid,color ="red",fill=True,fill_opacity=1,fill_color="red").add_to(folium_map)
                    folium.CircleMarker([path.matched_position_on_current_edge[1], path.matched_position_on_current_edge[0]],radius=1,popup=edgeid2,color ="red",fill=True,fill_opacity=1,fill_color="red").add_to(folium_map)
                else:
                    intersection_point = find_intersection(path.prev_path.edge.linestring, path.edge.linestring)
                    if intersection_point:
                        edgeid=((path.prev_path.edge.get_info()["start"],path.prev_path.edge.get_info()["end"]),path.prev_path.edge.get_info()["highway"])
                        edgeid2=((path.edge.get_info()["start"],path.edge.get_info()["end"]),path.edge.get_info()["highway"])
                        substring_line_prev = path.prev_path.extract_substring(path.prev_path.matched_position_on_current_edge, intersection_point)
                        substring_line_curr = path.extract_substring(intersection_point, path.matched_position_on_current_edge)
                        locations_prev = [(pt[1], pt[0]) for pt in substring_line_prev.coords]
                        locations_curr = [(pt[1], pt[0]) for pt in substring_line_curr.coords]
                        folium.PolyLine(locations=locations_prev,popup=edgeid,color=color).add_to(folium_map)
                        folium.PolyLine(locations=locations_curr,popup=edgeid2,color=color).add_to(folium_map)
                        folium.CircleMarker([path.matched_position_on_prev_edge[1], path.matched_position_on_prev_edge[0]],radius=1,popup=edgeid,color ="red",fill=True,fill_opacity=1,fill_color="red").add_to(folium_map)
                        folium.CircleMarker([path.matched_position_on_current_edge[1], path.matched_position_on_current_edge[0]],radius=1,popup=edgeid2,color ="red",fill=True,fill_opacity=1,fill_color="red").add_to(folium_map)
    for i,point in enumerate(trajectory):
        if i%4!=0:continue
        folium.CircleMarker([point[0], point[1]],radius=1.8,popup=i,color ="purple",fill=True,fill_opacity=1,fill_color="purple").add_to(folium_map)

    return folium_map



def find_intersection(linestring1, linestring2):
    intersection = linestring1.intersection(linestring2)
    if intersection.is_empty:
        return None
    else:
        return (intersection.x, intersection.y)





gps_traj = pd.read_csv('MyTracks/2023-06-10 22_33_53.csv')
df_nodes = pd.read_csv('GeoGraph/DriveWay_Node_Fukuoka_all_4000.csv')
df_edges = pd.read_csv('GeoGraph/DriveWay_Edge_Fukuoka_all_4000.csv',
                       usecols=['edge_id','start','end','length','linestring','highway','oneway','reversed','lanes','name'])


def write_paths_info_to_csv(final_k_best_paths, csv_file_path):
    with open(csv_file_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['prob', 'traj', 'current_path', 'matched_position_on_prev_edge', 'matched_position_on_current_edge'])

        for path_index, paths in enumerate(final_k_best_paths):
            writer.writerow([f"Path{path_index+1}"])
            for i,path in enumerate(paths):
                
                path_info = path.get_info()
                if i>-1:
                    writer.writerow([
                    path_info['prob'],
                    path_info['traj'],
                    (path_info['current_path'].get_info()["start"],path_info['current_path'].get_info()["end"]),
                    (path_info['matched_position_on_prev_edge'][1],path_info['matched_position_on_prev_edge'][0]),
                    (path_info['matched_position_on_current_edge'][1],path_info['matched_position_on_current_edge'][0])
                ])
            
import time
start_time = time.time()


graph = GeoGraph()
# Add nodes to the graph
for index, row in df_nodes.iterrows():
    graph.add_node(row['node_id'], row['lat'], row['lon'])
# Add edges to the graph
for index, row in df_edges.iterrows():
    graph.add_edge(row['edge_id'], row['start'], row['end'], row['length'], row['linestring'],row['highway'],row["oneway"],row["lanes"],row["name"])


gps_traj['geohash'] = gps_traj.apply(lambda row: geohash.encode(row['lat'], row['lon'], precision=7), axis=1)
trajectory = gps_traj[['lat', 'lon']].values.tolist()




matcher = MapMatcher(graph,7)

next_t = 0
i = 0
folium_map= folium.Map(location=[trajectory[0][0],trajectory[0][1]],tiles='cartodbpositron' ,zoom_start=20)
while next_t < len(trajectory):
    states, next_t = matcher.match(trajectory, pause=True, start=next_t)
    if not states:
        continue
    final_k_best_paths, _ = matcher.backtrack(states)
    filename = 'matched_path/paths_info_2023-06-10 22_33_53_{0}.csv'.format(i)  
    write_paths_info_to_csv(final_k_best_paths, filename)  
    folium_map=draw_paths(folium_map,final_k_best_paths)
    i += 1

  
folium_map.save("matched_traj/2023-06-10 22_33_53.html") 

end_time = time.time()
execution_time = end_time - start_time

print(execution_time)
