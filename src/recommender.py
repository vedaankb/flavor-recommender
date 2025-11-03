class HealthEffectRecommender:

    def __init__(self, flavor_engine, health_df, ingr_info_df):
        self.health_df = health_df
        self.ingr_info_df = ingr_info_df
        self.engine = flavor_engine

        # Column setup
        self.ing_col = "# ingredient id"
        self.effect_col = "health_effect"
        self.info_id_col = "# id"
        self.info_name_col = "ingredient name"

        # ✅ Lookup: ID → Name
        self.id_to_name_map = (
            ingr_info_df.set_index(self.info_id_col)[self.info_name_col].to_dict()
        )

    def id_to_name(self, ids):
        result = []
        for x in ids:
            # If already a valid ingredient name, keep it
            if isinstance(x, str) and x in set(self.id_to_name_map.values()):
                result.append(x)
            else:
                try:
                    xid = int(x)
                    result.append(self.id_to_name_map.get(xid, f"Unknown({xid})"))
                except:
                    result.append(f"Unknown({x})")
        return result


    def get_healthy_ingredients(self, effect_ids, top_n=10):
        """Return top N ingredient IDs linked to health effects"""
        matches = self.health_df[self.health_df[self.effect_col].isin(effect_ids)]
        if matches.empty:
            return None, "⚠️ No ingredients match these effects."

        scores = (
            matches.groupby(self.ing_col)
            .size()
            .reset_index(name="score")
            .sort_values("score", ascending=False)
            .head(top_n)
        )
        ids = scores[self.ing_col].tolist()
        names = self.id_to_name(ids)

        return ids, names

    def make_llm_recipe_prompt(self, effect_ids, top_n_pairs=5):
        """Generate full natural-language instruction prompt for another LLM"""

        # Get healthy ingredients
        ids, names = self.get_healthy_ingredients(effect_ids, top_n=15)
        if ids is None:
            return names  # error text

        # ✅ Use flavor network to find strong pairs among those ingredients
        ingredient_names = self.id_to_name(ids)
        best_pairs = self.engine.get_top_pairings(ingredient_names, top_k=top_n_pairs)


        if not best_pairs:
            return "⚠️ No flavor pairs found among selected healthy ingredients."

        # Convert pairs to readable names
        readable_pairs = []
        for i1, i2, score in best_pairs:
            name1, name2 = self.id_to_name([i1, i2])
            readable_pairs.append((name1, name2, score))

        # ✅ Structured Prompt Construction
        prompt_lines = [
            "You are a culinary AI. Create a recipe that:",
            "• Enhances these health effects: " +
            ", ".join(str(e) for e in effect_ids),
            "• MUST include at least one of these ingredient pairs:",
        ]

        for n1, n2, score in readable_pairs:
            prompt_lines.append(f"  - {n1} + {n2} (pair score {round(score,2)})")

        prompt_lines.append("\nRecipe requirements:")
        prompt_lines.append("• Unique dish name ✅")
        prompt_lines.append("• Brief reasoning why it supports the health effects ✅")
        prompt_lines.append("• Ingredients + Measurements ✅")
        prompt_lines.append("• Step-by-step cooking instructions ✅")
        prompt_lines.append("• Make it delicious and realistic ✅")

        return "\n".join(prompt_lines)