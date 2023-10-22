from geo_graph import GeoGraph
from path import Path
from edges import Edge
from relation import Relation
from shapely.geometry import LineString, Point
from shapely.ops import substring
from pyproj import CRS, Transformer
from geopy.distance import great_circle
from shapely import wkt
from shapely.ops import nearest_points
import math
import geohash
"""
"""
from bghash import Bghash
from scipy.special import logsumexp
import folium
from collections import defaultdict
import pandas as pd


epsilon = 1e-50
BETA=4.07
SIGMA_Z=3
MINIMUM=10**(-45)
MIN=10 ** (-95)

#gps_traj = pd.read_csv('MyTracks/2023-05-06_221836.csv')
#gps_traj['geohash'] = gps_traj.apply(lambda row: geohash.encode(row['lat'], row['lon'], precision=7), axis=1)
#trajectory = gps_traj[['lat', 'lon']].values.tolist()
#folium_map= folium.Map(location=[trajectory[0][0],trajectory[0][1]],tiles='cartodbpositron' ,zoom_start=20)


class MapMatcher:
    def __init__(self, gps_traj,graph, nodes, ways, relations,folium_map,k,sigma_z=4.07, beta=1.0):
        self.gps_traj=gps_traj
        self.graph = graph
        self.folium_map=folium_map
        self.nodes=nodes
        self.ways=ways
        self.relations=relations
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

    def calculate_distance_on_edge(self, edge, point):
        edge_shape = LineString(edge.linestring)
        edge_shape = LineString([(lat, lon) for lon, lat in edge_shape.coords])
        point_shape = Point(point)
        nearest_point_on_edge = nearest_points(edge_shape, point_shape)[0]
        point1 = (nearest_point_on_edge.x, nearest_point_on_edge.y)  
        point2 = (point_shape.x, point_shape.y)

        distance = great_circle(point1, point2).meters

        return distance, (nearest_point_on_edge.y, nearest_point_on_edge.x)
    
    
    def calculate_transition_probability(self, prev_edge, prev_traj, current_edge, current_traj):
        _, point_on_prev_edge = self.calculate_distance_on_edge(prev_edge, prev_traj)
        _, point_on_current_edge = self.calculate_distance_on_edge(current_edge, current_traj)
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
        intersection_node=self.graph.nodes[intersection_nodeid]
        routedis=self.calculate_routedistance(prev_edge, point_on_prev_edge, current_edge, point_on_current_edge,intersection_node)
        actual_distance = great_circle(prev_traj, current_traj).meters
        delta_distance = abs(routedis - actual_distance)

        delta_distance = abs(routedis - actual_distance)
        transition_prob = (1 / BETA) * math.exp(-delta_distance)

        return transition_prob
    
    def calculate_emission_probability(self, current_edge, current_traj):
        c = 1 / (SIGMA_Z * math.sqrt(2 * math.pi))
        distance_edge_traj,_=self.calculate_distance_on_edge(current_edge, current_traj)
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
        nodes_list = self.find_nodes_in_same_geohash(graph, gps_traj, start)

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
    

    def find_nodes_in_same_geohash(self, graph, gps_traj, index):
        """
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
        """
        #print(graph.bgeohash_dict)
        bgeohash_key = str(gps_traj.loc[index, 'geohash'])
        node_ids = []
        if bgeohash_key in graph.bgeohash_dict:
            node_ids += graph.bgeohash_dict[bgeohash_key]
        if len(node_ids) == 0:
            #neighbor_geohashes = geohash.neighbors(bgeohash_key)
            for i in range(8) :
                #print(i, Bghash(bgeohash_key).neighbor(i), Bghash(bgeohash_key).neighbor(i).decode())
                
                neighbor=str(bgeohash_key.neighbor(i))
                if neighbor in graph.bgeohash_dict:    
                    print("l;jl:;jk")        
                    node_ids += graph.bgeohash_dict[neighbor]   
        
        return node_ids
    
    def match(self, trajectory, pause=False, start=0):
        relation_switch=False
        transformer = Transformer.from_crs("epsg:4326", "epsg:32653")
        states = []
        i=0
        nodes_list = self.find_nodes_in_same_geohash(self.graph, self.gps_traj, start)
        candidate_ini_edgeslist = [edge for node in nodes_list for edge in self.graph.get_edges(node)]
        maxl=-1
        for edge in candidate_ini_edgeslist:
            if edge.highway=="tertiary" or edge.highway=="motorway_link" or edge.highway=="['motorway_link' 'tertiary']": continue
            dis,point= self.calculate_distance_on_edge(edge, trajectory[start])
            point1=self.calculate_emission_probability(edge, trajectory[start])
            point2=self.calculate_emission_probability(edge, trajectory[start+4])
            a=point1*point2#*point3
            if maxl<a and a>=1e-150:
                maxl=a
                nearest_edge=edge
                init_closest_point=point

        """
        初期化する際、近くにedgeがない場合すなわち（maxl <=1e-150）のとき、
        近くのリレーションを検索　→　リレーションに切り替えて、リレーションスイッチをオン

        """
        if maxl <=1e-150:
            
            print(start,"not any edge around traj")
            matching_ids= self.relations.get_relation_ids_by_geohash(trajectory[start][0],trajectory[start][1])


            try:
                edge_id, type, coords, nearest_point1=self.find_min_avg_dist_relation_and_nearest_point(matching_ids, self.ways, self.nodes, trajectory[start][0],trajectory[start][1])
                coords_transformed = [transformer.transform(*coord) for coord in coords]
                line = LineString(coords_transformed)
                traj1 = Point(transformer.transform(*(trajectory[start][0],trajectory[start][1])))
                point1 = transformer.transform(nearest_point1[0], nearest_point1[1])
                relation_switch=True  

            except TypeError: pass

        elif maxl >1e-150:
            print(start,"initial_state",maxl,(nearest_edge.get_info()["start"],nearest_edge.get_info()["end"]))

            iniprob = 1
        
            _,init_closest_point = self.calculate_distance_on_edge(nearest_edge, trajectory[start])
            states.append({
                nearest_edge: [Path(iniprob, trajectory[start],None,nearest_edge,init_closest_point,init_closest_point)]
            })
        else:
            return [],t+4
        for t in range(start+4, len(trajectory)):
            if t%4!=0:continue
            relation_emiprob=-1

            if relation_switch:
                
                traj2 = Point(transformer.transform(*(trajectory[t][0],trajectory[t][1])))
                nearest_point2 = nearest_points(LineString(coords), Point(trajectory[t][0],trajectory[t][1]))
                nearest_point2=(nearest_point2[0].x,nearest_point2[0].y)
                point2 = transformer.transform(nearest_point2[0], nearest_point2[1])
                proj_point1 = line.project(Point(point1))
                proj_point2 = line.project(Point(point2))
                route_distance = abs(proj_point1 - proj_point2)
                traj_distance=great_circle((trajectory[t-4][0],trajectory[t-4][1]),(trajectory[t][0],trajectory[t][1])).m
                dis=great_circle((nearest_point2[0], nearest_point2[1]),(trajectory[t][0],trajectory[t][1])).m
                c = 1 / (SIGMA_Z * math.sqrt(2 * math.pi))
                relation_emiprob=c * math.exp(-dis**2)
                print(t,relation_emiprob,route_distance,(nearest_point2,nearest_point1))
                dist1=LineString(coords).project(Point(trajectory[t-4][0],trajectory[t-4][1]))
                dist2=LineString(coords).project(Point(trajectory[t][0],trajectory[t][1]))
                cut_line = substring(LineString(coords), min(dist1, dist2), max(dist1, dist2))
                lat_lon_coords = [(lat, lon) for lat, lon in coords]
                folium.PolyLine(
                    locations=[(x, y) for x, y in list(cut_line.coords)],
                    color="pink"
                ).add_to(self.folium_map)
                folium.CircleMarker([nearest_point2[0], nearest_point2[1]],radius=1,popup="relation",color ="red",fill=True,fill_opacity=1,fill_color="red").add_to(self.folium_map)
                folium.CircleMarker([nearest_point1[0], nearest_point1[1]],radius=1,popup="relation",color ="red",fill=True,fill_opacity=1,fill_color="red").add_to(self.folium_map)
                point1=point2
                nearest_point1=nearest_point2
                if relation_emiprob<=1e-20: 
                    print(t,relation_emiprob)
                    return [],t
                continue
            if not states: return [],t
            prev_states = states[-1]
            current_states = defaultdict(list)  
            all_paths = [] 
            if not [self.calculate_emission_probability(self.graph.edges[state.get_info()["edge_id"]], trajectory[t-4]) for state in states[-1]]:
                print("V",t)
                return states[:-1], t

            if t>4 and max(self.calculate_emission_probability(self.graph.edges[state.get_info()["edge_id"]], trajectory[t-4]) for state in states[-1])<=1e-200:
                    for i in states[-1]:
                            print(t,"interrupt",(i.get_info()["start"],i.get_info()["end"]))
                    if not states[:-1]:
                        print("A",t)
                        return states[:-1], t+4
                    print("B",t)
                    return states[:-1], t-4
            for prev_edge in prev_states:
                if Edge.is_edge_instance(prev_edge)==False:   continue
                
                prev_edge_id = prev_edge.get_info()["edge_id"]
                start = self.graph.edges[prev_edge_id].start
                end = self.graph.edges[prev_edge_id].end
                candidate_edges = self.graph.get_adjacent_edges_from_nodes(start, end)
                candidate_edges = [edge for edge in candidate_edges if self.calculate_distance_on_edge(edge,trajectory[t])[0] < 30.0]
    
                if not candidate_edges: candidate_edges.append(prev_edge)

                for current_edge in candidate_edges:

                    if current_edge.highway=="motorway_link" or current_edge.highway=="tertiary": continue
                    for prev_path in prev_states[prev_edge]:
                        transition_prob = self.calculate_transition_probability(prev_edge, trajectory[t - 4], current_edge, trajectory[t])
                        emission_prob = self.calculate_emission_probability(current_edge, trajectory[t])
                        log_transition_prob = math.log(transition_prob + epsilon)
                        log_emission_prob = math.log(emission_prob + epsilon)
                        total_log_prob = self.log_prob_multiply(prev_path.log_prob, self.log_prob_multiply(log_transition_prob, log_emission_prob))
                        distance, prev_closest_point = self.calculate_distance_on_edge(prev_edge, trajectory[t - 4])
                        
                        distance, curr_closest_point = self.calculate_distance_on_edge(current_edge, trajectory[t])
                        new_path = Path(total_log_prob, trajectory[t],prev_path, current_edge, prev_closest_point,curr_closest_point)
                        
                        if new_path not in all_paths:
                            all_paths.append(new_path)
                            current_states[current_edge].append(new_path)
                            if len(current_states[current_edge]) > self.k:
                                diversity_penalty = 0.1  # 調整可能なハイパーパラメータ
                                current_states[current_edge].sort(key=lambda path: path.log_prob - diversity_penalty * min(self.calculate_diversity(path, p) for p in current_states[current_edge]), reverse=True)
                                current_states[current_edge] = current_states[current_edge][:self.k]
                                print(transition_prob*emission_prob)
            
            all_paths.sort(key=lambda path: path.log_prob, reverse=True)
            top_k_paths = all_paths[:self.k]

            next_states = defaultdict(list)
            
            for path in top_k_paths:
                next_states[path.edge].append(path)
            
            states.append(next_states)

        return states,len(trajectory)
    
    def calculate_diversity(self,path1, path2):
        min_length = min(len(path1.sequence), len(path2.sequence))
        matching_edges = sum(e1 == e2 for e1, e2 in zip(path1.sequence, path2.sequence))
        matching_ratio = matching_edges / min_length if min_length > 0 else 1
        diversity = 1 - matching_ratio  # 一致度が高いほど多様性は低いため、1から引く
        return diversity

    
    """
    一番近いリレーションを抜き出す
    """
    def find_min_avg_dist_relation_and_nearest_point(self, osmid_list, ways, nodes, lat, lon):
        min_dist = float('inf')  # 最小距離を保存する変数を初期化
        min_dist_osmid = None  # 最小距離に対応するosmidを保存する変数を初期化
        min_avg_dist_relation = None
        nearest_point = None
        point = Point(lat, lon)
        for osmid in osmid_list:
            relation = self.relations.get_relation_by_id(osmid)
            filtered_members = [member for member in relation['members'] if member[1] in {'inner', 'outer'} and member[2] == 'w' and 'alt_name' not in relation['tags']]
            if not filtered_members:
                continue

            for member in filtered_members:
                if member[2] == 'w': 
                    try:
                        way = ways[member[0]]
                        coords = [(nodes[node_id].lat, nodes[node_id].lon) for node_id in way.nodes if node_id in nodes]
                        line = LineString(coords)
                        result = nearest_points(line, point)
                        nearest_point_on_line = (result[0].x, result[0].y)
                        distance = great_circle((nearest_point_on_line[0], nearest_point_on_line[1]), (lat, lon)).meters
                        if member[1] == 'outer': 
                            if distance < min_dist:
                                min_dist = distance
                                min_dist_osmid = osmid
                                min_avg_dist_relation = relation
                                nearest_point = nearest_point_on_line
                                edgeid=member[0]
                    except KeyError as e:
                        pass
        rel=self.relations.get_relation_by_id(min_dist_osmid)
        if min_dist>50:
            return None,None,None,None
        
        way=ways[edgeid]
        coords = [(nodes[node_id].lat, nodes[node_id].lon) for node_id in way.nodes if node_id in nodes]
        return  edgeid, rel["type"],coords, nearest_point