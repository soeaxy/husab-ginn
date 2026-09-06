from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from manuscript_protocol import GCN_TRANSFORMER_CONFIG

DEEP_COMPARE_ALGOS = {"mlp", "gnn", "vae", "transunet", "gcn_transformer"}


def resolve_torch_device(device_arg: str = "auto") -> torch.device:
    requested = str(device_arg or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhysicsConstraintMixin:
    """Legacy optional-prior support for non-manuscript exploratory models."""

    def _init_physics_params(self) -> None:
        self.log_sigma_dome = nn.Parameter(torch.zeros(1))
        self.log_sigma_fault = nn.Parameter(torch.zeros(1))
        self.log_sigma_strata = nn.Parameter(torch.zeros(1))
        self.log_sigma_singularity = nn.Parameter(torch.zeros(1))

    def physics_loss(
        self,
        logits: torch.Tensor,
        prior_dome: torch.Tensor | None = None,
        prior_fault: torch.Tensor | None = None,
        prior_strata: torch.Tensor | None = None,
        prior_singularity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        terms: list[torch.Tensor] = []
        if prior_dome is not None:
            loss_dome = torch.mean((prob - prior_dome) ** 2)
            terms.append(
                0.5 * torch.exp(-2 * self.log_sigma_dome) * loss_dome
                + self.log_sigma_dome
            )
        if prior_fault is not None:
            loss_fault = torch.mean((prob - prior_fault) ** 2)
            terms.append(
                0.5 * torch.exp(-2 * self.log_sigma_fault) * loss_fault
                + self.log_sigma_fault
            )
        if prior_strata is not None:
            loss_strata = torch.mean((prob - prior_strata) ** 2)
            terms.append(
                0.5 * torch.exp(-2 * self.log_sigma_strata) * loss_strata
                + self.log_sigma_strata
            )
        if prior_singularity is not None:
            loss_singularity = torch.mean((prob - prior_singularity) ** 2)
            terms.append(
                0.5 * torch.exp(-2 * self.log_sigma_singularity) * loss_singularity
                + self.log_sigma_singularity
            )
        if not terms:
            return logits.new_tensor(0.0)
        return sum(terms)


class DataDrivenMLPClassifier(nn.Module):
    """Prior-loss-free MLP matched to the GINN classifier backbone."""

    def __init__(
        self, input_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="tanh")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class GNNFeatureClassifier(PhysicsConstraintMixin, nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self._init_physics_params()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAEClassifier(PhysicsConstraintMixin, nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._init_physics_params()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return logits, recon, mu, logvar

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return self.classifier(mu)


class TabularTransUNetClassifier(PhysicsConstraintMixin, nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._init_physics_params()
        if d_model % nhead != 0:
            d_model = max(nhead, (d_model // nhead) * nhead)
        self.input_dim = int(input_dim)
        self.token_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.input_dim, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.skip_fuse = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_embed[:, : x.shape[1], :]
        encoded = self.encoder(tokens)
        fused = self.skip_fuse(torch.cat([encoded, tokens], dim=-1))
        pooled = self.norm(fused.mean(dim=1))
        return self.head(pooled)


class GCNTransformerClassifier(nn.Module):
    """Inductive spatial GCN-Transformer over fixed-size local ego graphs.

    The first token is the query sample; the remaining tokens are its nearest
    training samples in projected map coordinates. The model uses no neighbor
    labels and no GINN geological-prior loss.
    """

    def __init__(
        self,
        input_dim: int,
        neighborhood_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.neighborhood_size = int(neighborhood_size)
        self.node_proj = nn.Linear(self.input_dim, d_model)
        self.gcn_self = nn.Linear(d_model, d_model, bias=False)
        self.gcn_neighbor = nn.Linear(d_model, d_model, bias=False)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.neighborhood_size + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(
                "GCN-Transformer expects [batch, centre-plus-neighbors, features] "
                f"with {self.input_dim} features; received {tuple(x.shape)}."
            )
        if x.shape[1] != self.neighborhood_size + 1:
            raise ValueError(
                f"Expected {self.neighborhood_size + 1} ego-graph nodes, received {x.shape[1]}."
            )
        tokens = self.node_proj(x)
        neighbor_mean = tokens[:, 1:, :].mean(dim=1)
        centre = torch.nn.functional.gelu(
            self.gcn_self(tokens[:, 0, :]) + self.gcn_neighbor(neighbor_mean)
        )
        tokens = tokens.clone()
        tokens[:, 0, :] = centre
        encoded = self.encoder(tokens + self.pos_embed)
        return self.head(self.norm(encoded[:, 0, :]))


def augment_with_knn_graph_features(
    X_ref: np.ndarray, X_query: np.ndarray, k_neighbors: int = 8
) -> np.ndarray:
    if X_query.size == 0:
        return X_query.astype(np.float32)
    if X_ref.size == 0:
        return np.concatenate([X_query, X_query], axis=1).astype(np.float32)
    k = int(max(1, min(k_neighbors, X_ref.shape[0])))
    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn_model.fit(X_ref)
    idx = nn_model.kneighbors(X_query, return_distance=False)
    neigh_mean = X_ref[idx].mean(axis=1)
    return np.concatenate([X_query, neigh_mean], axis=1).astype(np.float32)


def build_spatial_ego_graphs(
    X_ref: np.ndarray,
    coords_ref: np.ndarray,
    X_query: np.ndarray,
    coords_query: np.ndarray,
    k_neighbors: int = 8,
    exclude_self: bool = False,
) -> np.ndarray:
    """Build local spatial graphs without exposing labels across data splits."""
    X_ref = np.asarray(X_ref, dtype=np.float32)
    X_query = np.asarray(X_query, dtype=np.float32)
    coords_ref = np.asarray(coords_ref, dtype=np.float64)
    coords_query = np.asarray(coords_query, dtype=np.float64)
    if X_ref.ndim != 2 or X_query.ndim != 2 or X_ref.shape[1] != X_query.shape[1]:
        raise ValueError(
            "Reference and query feature matrices must be 2-D with equal feature dimensions."
        )
    if coords_ref.shape != (X_ref.shape[0], 2) or coords_query.shape != (
        X_query.shape[0],
        2,
    ):
        raise ValueError("Coordinate matrices must have shape [n_samples, 2].")
    requested = int(k_neighbors) + int(exclude_self)
    if requested < 1 or X_ref.shape[0] <= requested:
        raise ValueError(
            "Reference set is too small for the requested spatial neighborhood size."
        )
    knn = NearestNeighbors(n_neighbors=requested, metric="euclidean").fit(coords_ref)
    indices = knn.kneighbors(coords_query, return_distance=False)
    if exclude_self:
        if X_ref.shape[0] != X_query.shape[0] or not np.array_equal(
            coords_ref, coords_query
        ):
            raise ValueError(
                "exclude_self=True requires identical query and reference sets."
            )
        row_ids = np.arange(X_query.shape[0])[:, None]
        indices = indices[indices != row_ids].reshape(
            X_query.shape[0], int(k_neighbors)
        )
    return np.concatenate([X_query[:, None, :], X_ref[indices]], axis=1).astype(
        np.float32, copy=False
    )


def build_deep_compare_model(
    algorithm: str, input_dim: int
) -> tuple[nn.Module, dict[str, Any]]:
    algo = algorithm.lower()
    if algo == "mlp":
        return DataDrivenMLPClassifier(input_dim=input_dim), {
            "input_dim": input_dim,
            "backbone": "ginn_matched",
        }
    if algo == "gnn":
        return GNNFeatureClassifier(input_dim=input_dim, hidden_dim=128, dropout=0.2), {
            "input_dim": input_dim
        }
    if algo == "vae":
        return VAEClassifier(
            input_dim=input_dim, hidden_dim=128, latent_dim=32, dropout=0.1
        ), {"input_dim": input_dim}
    if algo == "transunet":
        return (
            TabularTransUNetClassifier(
                input_dim=input_dim,
                d_model=64,
                nhead=8,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
            ),
            {"input_dim": input_dim},
        )
    if algo == "gcn_transformer":
        config = GCN_TRANSFORMER_CONFIG
        return (
            GCNTransformerClassifier(
                input_dim=input_dim,
                neighborhood_size=int(config["neighborhood_size"]),
                d_model=int(config["d_model"]),
                nhead=int(config["nhead"]),
                num_layers=int(config["num_layers"]),
                dim_feedforward=int(config["dim_feedforward"]),
                dropout=float(config["dropout"]),
            ),
            {
                "input_dim": input_dim,
                "neighborhood_size": int(config["neighborhood_size"]),
                "d_model": int(config["d_model"]),
                "nhead": int(config["nhead"]),
                "num_layers": int(config["num_layers"]),
                "dim_feedforward": int(config["dim_feedforward"]),
                "dropout": float(config["dropout"]),
                "position_tokens": int(config["position_tokens"]),
                "graph": "inductive_spatial_knn_mean_aggregation",
                "neighbor_labels_used": False,
            },
        )
    raise ValueError(f"Unsupported deep comparison algorithm: {algorithm}")


def load_deep_compare_model(
    algorithm: str, checkpoint: dict[str, Any], device: torch.device
) -> nn.Module:
    meta = checkpoint.get("model_meta") or {}
    input_dim = int(meta.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError("Invalid model_meta.input_dim in deep model checkpoint.")
    model, _ = build_deep_compare_model(algorithm, input_dim=input_dim)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Deep model checkpoint missing model_state.")
    model.load_state_dict(state)
    return model.to(device)


def predict_probabilities_deep_compare(
    model: nn.Module,
    x_data: np.ndarray,
    batch_size: int,
    device: torch.device,
    algorithm: str,
) -> np.ndarray:
    algo = algorithm.lower()
    dataset = TensorDataset(torch.tensor(x_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model.predict_logits(xb) if algo == "vae" else model(xb)
            prob = (
                torch.softmax(logits, dim=1)[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            probs.append(prob)
    return np.concatenate(probs) if probs else np.empty((0,), dtype=np.float32)


def predict_probabilities_estimator(model: Any, x_data: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x_data), dtype=np.float64)
        if proba.ndim == 1:
            return proba.astype(np.float32)
        if proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        return proba[:, 0].astype(np.float32)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x_data), dtype=np.float64)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)
    raise ValueError("Loaded estimator does not support probability output.")
