import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations

class FlavorNetworkEngine:
    def __init__(self, graph_path, bridging_scores_path=None):
        print("Loading flavor network...")
        self.G = nx.read_gml(graph_path)
        print(f"✅ Network loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        # Load bridging scores if provided
        if bridging_scores_path:
            self.bridging_df = pd.read_csv(bridging_scores_path)
            self.bridging_scores = dict(zip(self.bridging_df['ingredient'], self.bridging_df['bridging_score']))
        else:
            self.bridging_df = None
            self.bridging_scores = {}

        # Normalize edge weights between 0 and 1 for scaling later
        weights = [d['weight'] for _, _, d in self.G.edges(data=True)]
        self.max_weight = max(weights)
        self.min_weight = min(weights)
        print(f"✅ Weight range: {self.min_weight} - {self.max_weight}")

    def get_pair_score(self, ingr1, ingr2):
        ingr1 = ingr1.lower().strip()
        ingr2 = ingr2.lower().strip()

        if ingr1 not in self.G or ingr2 not in self.G:
            return {"error": f"One or both ingredients not found: '{ingr1}', '{ingr2}'"}

        # Case 1: Direct edge exists — scale to 0–100
        if self.G.has_edge(ingr1, ingr2):
            w = self.G[ingr1][ingr2]['weight']
            norm_score = (w - self.min_weight) / (self.max_weight - self.min_weight)
            scaled_score = round(norm_score * 100, 2)  # out of 100
            return {
                "ingredients": (ingr1, ingr2),
                "pairing_type": "direct",
                "pair_score": scaled_score,
                "bridge_ingredient": None
            }

        # Case 2: No direct link → find a bridge path
        try:
            path = nx.shortest_path(self.G, ingr1, ingr2)
            if len(path) > 2:
                bridge = path[1:-1]
            else:
                bridge = None

            # If bridging scores exist, pick the best bridge
            best_bridge = None
            if bridge and self.bridging_scores:
                best_bridge = max(bridge, key=lambda x: self.bridging_scores.get(x, 0))

            # Assign a low pairing score, scaled to 0–100
            scaled_score = round(10 + np.random.uniform(0, 5), 2)  # weak link ~10–15/100

            return {
                "ingredients": (ingr1, ingr2),
                "pairing_type": "bridged",
                "pair_score": scaled_score,
                "bridge_ingredient": best_bridge
            }

        except nx.NetworkXNoPath:
            return {
                "ingredients": (ingr1, ingr2),
                "pairing_type": "disconnected",
                "pair_score": 0.0,
                "bridge_ingredient": None
            }
    def get_top_pairings(self, ingredient_names, top_k=5):
        """
        Given a list of ingredient NAMES, rank best (direct or bridged) pairings.
        Returns: list of (name1, name2, score)
        """
        results = []

        # Generate all possible ingredient pairs
        for ingr1, ingr2 in combinations(ingredient_names, 2):
            pair_info = self.get_pair_score(ingr1, ingr2)

            if "pair_score" in pair_info and pair_info["pair_score"] > 0:
                results.append((ingr1, ingr2, pair_info["pair_score"]))

        if not results:
            return []

        # Sort best → worst
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:top_k]