from nodes import Nodes
from edges import Edge
from path import Path
from geo_graph import GeoGraph
from map_matcher import MapMatcher
from relation import Relation
import pandas as pd
import csv
import ast
import geohash
"""
"""
from bghash import Bghash
import time
import folium
from osmread import parse_file, Node, Way



def draw_paths(folium_map,paths):

    def find_intersection(linestring1, linestring2):
        intersection = linestring1.intersection(linestring2)
        if intersection.is_empty:
            return None
        else:
            return (intersection.x, intersection.y)
        
    for i,point in enumerate(trajectory):
        if i%4!=0:continue
        folium.CircleMarker([point[0], point[1]],radius=1.8,popup=i,color ="purple",fill=True,fill_opacity=1,fill_color="purple").add_to(folium_map)
   
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

    return folium_map


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


def process_osm_file_to_relations(file_path, relation_csv_path):
    nodes = {}
    ways = {}

    for entity in parse_file(file_path):
        if isinstance(entity, Node):
            nodes[entity.id] = entity
        elif isinstance(entity, Way):
            ways[entity.id] = entity

    relations = Relation(nodes, ways)
    df = pd.read_csv(relation_csv_path)

    for index, row in df.iterrows():
        members = ast.literal_eval(row['members'])
        tags = ast.literal_eval(row['tags'])
        relations.add_relation(row['osmid'], members, tags)
    
    return nodes, ways, relations



gps_traj = pd.read_csv('mapmatch/MyTracks/2023-05-06_221836.csv')
df_nodes = pd.read_csv('mapmatch/GeoGraph/DriveWay_Node_Fukuoka_all_4000.csv')
df_edges = pd.read_csv('mapmatch/GeoGraph/DriveWay_Edge_Fukuoka_all_4000.csv',
                        usecols=['edge_id','start','end','length','linestring','highway','oneway','reversed','lanes','name'])

# Add relation_network to the relations
nodes, ways, relations = process_osm_file_to_relations('mapmatch/GeoGraph/Fukuoka.osm', 'mapmatch/GeoGraph/Fukuoka_relations.csv')

graph = GeoGraph()
# Add nodes to the graph
for index, row in df_nodes.iterrows():
    graph.add_node(row['node_id'], row['lat'], row['lon'])
# Add edges to the graph
for index, row in df_edges.iterrows():
    graph.add_edge(row['edge_id'], row['start'], row['end'], row['length'], row['linestring'],row['highway'],row["oneway"],row["lanes"],row["name"])


gps_traj['geohash'] = gps_traj.apply(lambda row: geohash.encode(row['lat'], row['lon'], precision=7), axis=1)
"""
"""
gps_traj['geohash'] = gps_traj.apply(lambda row: Bghash(row['lat'], row['lon'], 32), axis=1)
trajectory = gps_traj[['lat', 'lon']].values.tolist()






def main():
    folium_map= folium.Map(location=[trajectory[0][0],trajectory[0][1]],tiles='cartodbpositron' ,zoom_start=20)

    matcher = MapMatcher(gps_traj,graph, nodes, ways, relations,folium_map,7)

    next_t = 0
    i = 0

    hashcode = Bghash(33.5925135, 130.3560714, 32)
    # e6f5da1cc0000025
   # print('hello.', (33.5925135, 130.3560714, 37), hashcode, hashcode.precision())
    hashcode = Bghash(0xe6f5da1dc8000025)
  #  print('from 0xe6f5da1dc8000025 ', hashcode)
   # print()
    
    hashcode = 'wvuxn7f'
    hashcode = Bghash(hashcode)
   # print('wvuxn7f', hashcode, hashcode.geohash(), hashcode.decode())
    
   # print(hashcode.decode())
    #print(hashcode.geohash())
    
    #print('ne', hashcode.neighbor('ne').decode())
    #print()
    for i in range(8) :
        print(i, hashcode,hashcode.neighbor(i))#, hashcode.neighbor(i).decode())
    


    start_time = time.time()
    while next_t < len(trajectory):
        states, next_t = matcher.match(trajectory, pause=True, start=next_t)
        if not states:
            print("next_t→",next_t)
            print("")
            continue
        final_k_best_paths, _ = matcher.backtrack(states)
        filename = 'mapmatch/matched_path/paths_info_2023-05-06_221836__{0}.csv'.format(i)  
        write_paths_info_to_csv(final_k_best_paths, filename)  
        folium_map=draw_paths(folium_map,final_k_best_paths)
        i += 1
    folium_map.save("mapmatch/matched_traj/2023-05-06_221836_.html") 


if __name__ == "__main__":
    main()