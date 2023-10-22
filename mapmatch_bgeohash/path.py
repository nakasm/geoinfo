from edges import Edge
from shapely.ops import substring
from shapely.geometry import LineString, Point

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