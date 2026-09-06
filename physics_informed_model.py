from __future__ import annotations

import torch
import torch.nn as nn


class GeologyInformedClassifier(nn.Module):
    """GINN classifier with uncertainty-weighted geological prior losses.

    Dome, fault and stratigraphic scores are the three priors used by the
    manuscript experiments.  ``log_sigma_singularity`` remains registered only
    so archived checkpoints and the optional legacy input stay loadable; it is
    inactive unless a caller explicitly supplies a singularity prior.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        *,
        fixed_prior_weights: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 2)

        # Three manuscript priors plus one optional legacy checkpoint parameter.
        self.log_sigma_dome = nn.Parameter(torch.zeros(1))
        self.log_sigma_fault = nn.Parameter(torch.zeros(1))
        self.log_sigma_strata = nn.Parameter(torch.zeros(1))
        self.log_sigma_singularity = nn.Parameter(torch.zeros(1))
        if fixed_prior_weights:
            for name in ("dome", "fault", "strata", "singularity"):
                getattr(self, f"log_sigma_{name}").requires_grad_(False)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="tanh")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def geological_prior_loss(
        self,
        logits: torch.Tensor,
        prior_dome: torch.Tensor | None = None,
        prior_fault: torch.Tensor | None = None,
        prior_strata: torch.Tensor | None = None,
        prior_singularity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Uncertainty-weighted geological-prior loss:
        L = sum_i [0.5 * exp(-2*log_sigma_i) * mse_i + log_sigma_i]
        """
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

    def physics_loss(
        self,
        logits: torch.Tensor,
        prior_dome: torch.Tensor | None = None,
        prior_fault: torch.Tensor | None = None,
        prior_strata: torch.Tensor | None = None,
        prior_singularity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible name for :meth:`geological_prior_loss`."""

        return self.geological_prior_loss(
            logits,
            prior_dome,
            prior_fault,
            prior_strata,
            prior_singularity,
        )


# Archived code and checkpoints use this class name.  Keep it as an alias while
# exposing terminology that matches the manuscript and the implemented method.
PhysicsInformedClassifier = GeologyInformedClassifier
