from src.flavor_engine import FlavorNetworkEngine
from src.recommender import HealthEffectRecommender
import pandas as pd

print("✅ Starting Test Script...")

# --- Load data ---
engine = FlavorNetworkEngine(
    "data/ingredient_network_weighted_fixed.gml",
    "data/ingredient_bridging_scores.csv"
)

ingr_df = pd.read_csv("data/ingr_info.tsv", sep="\t")
health_df = pd.read_csv("data/compound_ingredient_health_mapping.tsv", sep="\t")

# Print headers for debug
print("\nIngredient CSV Columns:", ingr_df.columns.tolist())
print("Health CSV Columns:", health_df.columns.tolist())

# --- Create recommender ---
recommender = HealthEffectRecommender(engine, health_df, ingr_df)

# ✅ Test Example: target health effects
effects = [10, 25]

# ✅ Step 1: get healthy ingredients
ids, names = recommender.get_healthy_ingredients(effects, top_n=10)
print("\n✅ Healthy Ingredient Results:")
print(names)

# ✅ Step 2: get top flavor pairs
pairs = engine.get_top_pairings(names, top_k=5)
print("\n✅ Top Flavor Pairings:")
for a, b, score in pairs:
    print(f"{a} + {b} → score: {score}")

# ✅ Step 3: Generate recipe LLM prompt
prompt = recommender.make_llm_recipe_prompt(effects, top_n_pairs=5)
print("\n✅ Generated LLM Prompt:\n")
print(prompt)
