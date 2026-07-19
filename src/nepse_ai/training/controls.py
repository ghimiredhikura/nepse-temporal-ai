"""Convergence controls for chronological neural experiments."""

from __future__ import annotations

import math

import torch


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay, stepped once per epoch."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        minimum_learning_rate: float,
    ) -> None:
        if not 0 <= warmup_epochs < total_epochs:
            raise ValueError("warmup_epochs must be below total_epochs")
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.minimum_learning_rate = minimum_learning_rate
        self.base_rates = [
            group["lr"] for group in optimizer.param_groups
        ]

    def step(self, epoch: int) -> list[float]:
        if epoch <= self.warmup_epochs and self.warmup_epochs:
            factor = epoch / self.warmup_epochs
            rates = [base * factor for base in self.base_rates]
        else:
            progress = (
                epoch - self.warmup_epochs
            ) / max(1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            rates = [
                self.minimum_learning_rate
                + (base - self.minimum_learning_rate) * cosine
                for base in self.base_rates
            ]
        for group, rate in zip(
            self.optimizer.param_groups, rates, strict=True
        ):
            group["lr"] = rate
        return rates


class EarlyStopping:
    """Stop only after minimum training and sustained non-improvement."""

    def __init__(
        self,
        patience: int,
        minimum_epochs: int,
        minimum_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.minimum_epochs = minimum_epochs
        self.minimum_delta = minimum_delta
        self.best = float("-inf")
        self.bad_epochs = 0

    def update(self, epoch: int, score: float) -> bool:
        if score > self.best + self.minimum_delta:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return (
            epoch >= self.minimum_epochs
            and self.bad_epochs >= self.patience
        )
