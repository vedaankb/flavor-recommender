from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from src.flavor_engine import FlavorNetworkEngine
from src.recommender import HealthEffectRecommender


# -----------------------------
# App and CORS configuration
# -----------------------------
app = FastAPI(title="Flavor Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("flavor_recommender")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# -----------------------------
# Global resources (lazy-loaded)
# -----------------------------
engine: Optional[FlavorNetworkEngine] = None
recommender: Optional[HealthEffectRecommender] = None


def load_resources() -> None:
    global engine, recommender

    logger.debug("load_resources called")
    if engine is not None and recommender is not None:
        logger.debug("Resources already loaded; skipping initialization")
        return

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    graph_path = data_dir / "ingredient_network_weighted_fixed.gml"
    bridging_scores_path = data_dir / "ingredient_bridging_scores.csv"
    ingr_info_path = data_dir / "ingr_info.tsv"
    health_map_path = data_dir / "compound_ingredient_health_mapping.tsv"

    engine = FlavorNetworkEngine(
        str(graph_path),
        str(bridging_scores_path),
    )

    ingr_df = pd.read_csv(str(ingr_info_path), sep="\t")
    health_df = pd.read_csv(str(health_map_path), sep="\t")

    recommender = HealthEffectRecommender(engine, health_df, ingr_df)
    logger.info(
        "Resources initialized (graph=%s, bridging=%s, ingr=%s, health=%s)",
        graph_path.name,
        bridging_scores_path.name,
        ingr_info_path.name,
        health_map_path.name,
    )


@app.on_event("startup")
def on_startup() -> None:
    load_resources()


# -----------------------------
# Request/Response models
# -----------------------------
class EffectsRequest(BaseModel):
    effects: List[int] = Field(..., description="Health effect IDs")
    top_n: int = Field(10, ge=1, le=100, description="Number of ingredients to return")


class PairingsRequest(BaseModel):
    ingredient_names: List[str] = Field(..., description="List of ingredient names")
    top_k: int = Field(5, ge=1, le=100, description="Number of pairs to return")


class PromptRequest(BaseModel):
    effects: List[int] = Field(..., description="Health effect IDs")
    top_n_pairs: int = Field(5, ge=1, le=100, description="Number of pairs to include in prompt")


class CombinedRequest(BaseModel):
    effects: List[int]
    top_n_ingredients: int = Field(10, ge=1, le=100)
    top_k_pairs: int = Field(5, ge=1, le=100)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health() -> dict:
    load_resources()
    logger.debug("Health check")
    return {"status": "ok"}


@app.post("/recommend/ingredients")
def recommend_ingredients(payload: EffectsRequest) -> dict:
    load_resources()
    assert recommender is not None
    logger.info(
        "POST /recommend/ingredients effects=%s top_n=%d",
        payload.effects,
        payload.top_n,
    )

    ids, names = recommender.get_healthy_ingredients(payload.effects, top_n=payload.top_n)
    if ids is None:
        logger.warning("No ingredients found for effects=%s", payload.effects)
        raise HTTPException(status_code=404, detail=names)
    return {"ingredient_ids": ids, "ingredient_names": names}


@app.post("/recommend/pairings")
def recommend_pairings(payload: PairingsRequest) -> dict:
    load_resources()
    assert engine is not None
    logger.info(
        "POST /recommend/pairings ingredient_names=%s top_k=%d",
        payload.ingredient_names,
        payload.top_k,
    )

    pairs = engine.get_top_pairings(payload.ingredient_names, top_k=payload.top_k)
    # pairs: List[Tuple[str, str, float]] → return as objects
    formatted = [
        {"ingredient_a": a, "ingredient_b": b, "pair_score": float(score)}
        for a, b, score in pairs
    ]
    return {"pairs": formatted}


@app.post("/recommend/prompt")
def generate_prompt(payload: PromptRequest) -> dict:
    load_resources()
    assert recommender is not None
    logger.info(
        "POST /recommend/prompt effects=%s top_n_pairs=%d",
        payload.effects,
        payload.top_n_pairs,
    )

    prompt = recommender.make_llm_recipe_prompt(payload.effects, top_n_pairs=payload.top_n_pairs)
    if prompt.startswith("⚠️"):
        logger.warning("Prompt generation failed for effects=%s: %s", payload.effects, prompt)
        raise HTTPException(status_code=404, detail=prompt)
    return {"prompt": prompt}


@app.post("/recommend")
def full_recommendation(payload: CombinedRequest) -> dict:
    load_resources()
    assert recommender is not None and engine is not None
    logger.info(
        "POST /recommend effects=%s top_n_ingredients=%d top_k_pairs=%d",
        payload.effects,
        payload.top_n_ingredients,
        payload.top_k_pairs,
    )

    ids, names = recommender.get_healthy_ingredients(payload.effects, top_n=payload.top_n_ingredients)
    if ids is None:
        logger.warning("No ingredients found for effects=%s", payload.effects)
        raise HTTPException(status_code=404, detail=names)

    pairs = engine.get_top_pairings(names, top_k=payload.top_k_pairs)
    pairs_formatted = [
        {"ingredient_a": a, "ingredient_b": b, "pair_score": float(score)}
        for a, b, score in pairs
    ]

    prompt = recommender.make_llm_recipe_prompt(payload.effects, top_n_pairs=payload.top_k_pairs)

    return {
        "ingredient_ids": ids,
        "ingredient_names": names,
        "pairs": pairs_formatted,
        "prompt": prompt,
    }


# To run locally: uvicorn app:app --reload

