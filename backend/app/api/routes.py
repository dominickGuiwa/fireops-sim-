"""Version 0.1 REST API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app import __version__
from app.models.schemas import (
    ChaosRequest,
    ConfigUpdate,
    HealthResponse,
    KnowledgeResponse,
    LearningStatsResponse,
    PoliciesResponse,
    ResetRequest,
    StateResponse,
    StepRequest,
    TrainRequest,
    TrainResponse,
)
from app.services.knowledge import answer_query
from app.services.session import SimulationSession


router = APIRouter(prefix="/api")
session = SimulationSession()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="fireops-sim", version=__version__)


@router.get("/state", response_model=StateResponse, tags=["simulation"])
def get_state() -> StateResponse:
    return session.state()


@router.post("/reset", response_model=StateResponse, tags=["simulation"])
def reset(request: ResetRequest | None = None) -> StateResponse:
    request = request or ResetRequest()
    try:
        return session.reset(
            seed=request.seed,
            policy=request.policy,
            config_update=request.config,
            sensor_reliability=request.sensor_reliability,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/step", response_model=StateResponse, tags=["simulation"])
def step(request: StepRequest | None = None) -> StateResponse:
    request = request or StepRequest()
    try:
        return session.step(action=request.action, policy=request.policy)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/chaos", response_model=StateResponse, tags=["simulation"])
def chaos(request: ChaosRequest | None = None) -> StateResponse:
    request = request or ChaosRequest()
    return session.chaos(request.failure)


@router.put("/config", response_model=StateResponse, tags=["simulation"])
def update_config(request: ConfigUpdate) -> StateResponse:
    try:
        return session.update_config(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/policies", response_model=PoliciesResponse, tags=["agents"])
def get_policies() -> PoliciesResponse:
    return session.policies()


@router.get("/learning", response_model=LearningStatsResponse, tags=["agents"])
def get_learning_stats() -> LearningStatsResponse:
    return LearningStatsResponse.model_validate(session.learning_stats())


@router.post("/train", response_model=TrainResponse, tags=["agents"])
def train(request: TrainRequest | None = None) -> TrainResponse:
    request = request or TrainRequest()
    return TrainResponse.model_validate(
        session.train(
            episodes=request.episodes,
            seed=request.seed,
            difficulty=request.difficulty,
        )
    )


@router.get("/knowledge", response_model=KnowledgeResponse, tags=["knowledge"])
def knowledge(
    query: str = Query(min_length=2, max_length=200),
    k: int = Query(default=3, ge=1, le=5),
) -> KnowledgeResponse:
    """Retrieval-augmented lookup in the synthetic simulator field manual."""

    return KnowledgeResponse.model_validate(answer_query(query, k=k))


@router.post("/policies/register", tags=["agents"], status_code=status.HTTP_501_NOT_IMPLEMENTED)
def register_policy() -> None:
    """Prototype endpoint reserved for future external policy plugins."""

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Prototype: external policy registration is planned but intentionally not "
            "implemented yet. Today a new AI ships in-repo by implementing the Agent "
            "protocol (act(observation) -> int) in backend/app/agents and wiring it into "
            "app/services/session.py; saved PPO/DQN artifacts load through environment "
            "variables. The simulator keeps running normally after this call."
        ),
    )
