from typing import Optional

import torch
import torch.nn as nn
import zuko


def replace_linear(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module)

        if isinstance(module, zuko.nn.Linear):
            new_linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_linear.weight = module.weight
            if module.bias is not None:
                new_linear.bias = module.bias
            setattr(model, n, new_linear)


class Flow(nn.Module):
    flow: zuko.flows.LazyDistribution

    def __init__(self, data_dim: int, context_dim: int):
        super().__init__()
        self.data_dim = data_dim
        self.context_dim = context_dim

    def condition(
        self, context: Optional[torch.Tensor] = None, null_context: Optional[torch.Tensor] = None, cfg_w: float = 1.0
    ):
        if self.context_dim == 0 and context is None:
            return self.flow()

        else:
            if cfg_w == 0.0:
                assert null_context is not None
                return self.flow(null_context)
            elif cfg_w == 1.0:
                return self.flow(context)
            else:
                context = (1 - cfg_w) * null_context + cfg_w * context
                return self.flow(context)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
    ):
        return self.condition(context, null_context=null_context, cfg_w=cfg_w).log_prob(x)

    def sample(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
        **kwargs,
    ):
        return self.condition(context, null_context=null_context, cfg_w=cfg_w).sample((num_samples,))

    def step(self, batch, batch_idx=None, **kwargs):
        if len(batch) == 2:
            x, context = batch
            return -self.log_prob(x, context=context).mean()
        # unconditional
        return -self.log_prob(batch).mean()  # minimize negative log likelihood over all elements


class NSF(Flow):
    def __init__(
        self,
        data_dim: int,
        context_dim: int = 0,
        layers: int = 4,
        bins: int = 8,
        hidden_dims: list[int] = [64, 64],
        activation: str = "GELU",
        rescale: float = 1.0,
    ):
        super().__init__(data_dim, context_dim)
        self.flow = zuko.flows.NSF(
            features=data_dim,
            context=context_dim,
            bins=bins,
            transforms=layers,
            hidden_features=hidden_dims,
            activation=getattr(torch.nn, activation),
        )
        self.rescale = rescale
        replace_linear(self.flow)

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
    ):
        return self.condition(context, null_context=null_context, cfg_w=cfg_w).log_prob(x * self.rescale)

    def sample(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
        null_context: Optional[torch.Tensor] = None,
        cfg_w: float = 1.0,
        **kwargs,
    ):
        return self.condition(context, null_context=null_context, cfg_w=cfg_w).sample((num_samples,)) / self.rescale


class RealNVP(Flow):
    def __init__(
        self,
        data_dim: int,
        context_dim: int = 0,
        layers: int = 4,
        hidden_dims: list[int] = [64, 64],
        activation: str = "ReLU",
    ):
        super().__init__(data_dim, context_dim)
        self.flow = zuko.flows.NICE(
            features=data_dim,
            context=context_dim,
            transforms=layers,
            hidden_features=hidden_dims,
            activation=getattr(torch.nn, activation),
        )
        replace_linear(self.flow)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if m.__class__.__name__ == "MLP":
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)


class GMM(Flow):
    def __init__(
        self,
        data_dim: int,
        context_dim: int = 0,
        components: int = 16,
        hidden_dims: list[int] = [64, 64],
        activation: str = "GELU",
    ):
        super().__init__(data_dim, context_dim)
        self.flow = zuko.flows.GMM(
            features=data_dim,
            context=context_dim,
            components=components,
            hidden_features=hidden_dims,
            activation=getattr(torch.nn, activation),
        )
        replace_linear(self.flow)

