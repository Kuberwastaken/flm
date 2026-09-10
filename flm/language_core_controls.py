"""Prospective language computation controls; no trainer or experiment is launched.

Keep these definitions separate from the frozen topology study. In particular,
the old trainer cannot restore the extended configuration as an ordinary FLM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Config, FLM, load_graph


CONTROLS = ('full', 'fixed_dynamics', 'no_lateral', 'no_temporal_state')
FROZEN_DYNAMICS = ('edge_log_gain', 'alpha_logit', 'beta_logit', 'recurrent_logit')


@dataclass
class CoreControlConfig(Config):
    control: str = 'full'


class LanguageCoreControl(FLM):
    def __init__(self, graph, config: CoreControlConfig):
        if not isinstance(config, CoreControlConfig) or config.control not in CONTROLS:
            raise ValueError('An explicit language core control configuration is required')
        expected = 'no_recurrence' if config.control in ('no_lateral', 'no_temporal_state') else 'flm'
        if config.variant != expected:
            raise ValueError('The control and underlying recurrence configuration disagree')
        super().__init__(graph, config)
        self._frozen_initial = {}
        if config.control == 'fixed_dynamics':
            for name in FROZEN_DYNAMICS:
                parameter = self.get_parameter(name)
                parameter.requires_grad_(False)
                self._frozen_initial[name] = parameter.detach().cpu().clone()

    def transition(self, drive, state, constants):
        if self.config.control == 'no_temporal_state':
            # Reset both states for EACH token, including within a training chunk.
            # Wh is identically zero at a fresh state, so no_lateral also removes
            # that ineffective multiplication from the autograd graph.
            state = self.initial_state(drive.shape[0])
        return super().transition(drive, state, constants)

    def verify_frozen_dynamics(self, state_dict=None):
        """Check exact initial tensors before saving and before loading a checkpoint.

        A future restore must construct with the declared original seed, verify
        the supplied state dictionary, and only then load model/optimizer state.
        Checking requires_grad alone would miss a corrupted saved frozen tensor.
        """
        if self.config.control != 'fixed_dynamics':
            raise ValueError('Frozen-dynamics verification requires the fixed control')
        values = self.state_dict() if state_dict is None else state_dict
        for name, expected in self._frozen_initial.items():
            if self.get_parameter(name).requires_grad:
                raise ValueError('Frozen parameter was made trainable: ' + name)
            actual = values.get(name)
            if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                    or not torch.equal(actual.detach().cpu(), expected)):
                raise ValueError('Frozen parameter differs from initialization: ' + name)

    def parameter_card(self):
        card = super().parameter_card()
        # The inherited card counts every stored parameter as trainable. Correct
        # that distinction here without changing the frozen model implementation.
        card['allocated_parameters'] = sum(p.numel() for p in self.parameters())
        card['trainable_parameters'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        card['frozen_parameters'] = card['allocated_parameters'] - card['trainable_parameters']
        card['frozen_groups'] = [name for name, p in self.named_parameters() if not p.requires_grad]
        return card

    def gradient_probe(self, loss):
        """Describe connectivity for one caller-supplied loss, without editing .grad.

        A connected zero derivative is different from an unused tensor. Neither
        observed nonzero entries nor this one probe measure model capacity.
        autograd.grad consumes this loss graph; use a fresh forward for training.
        """
        parameters = [(name, p) for name, p in self.named_parameters() if p.requires_grad]
        gradients = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
        groups = {}
        for (name, parameter), gradient in zip(parameters, gradients):
            if gradient is not None and not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite probe gradient: ' + name)
            groups[name] = dict(entries=parameter.numel(), connected=gradient is not None,
                                nonzero_entries=0 if gradient is None else int(torch.count_nonzero(gradient)))
        return dict(groups=groups,
                    connected_in_probe=sum(r['entries'] for r in groups.values() if r['connected']),
                    nonzero_in_probe=sum(r['nonzero_entries'] for r in groups.values()),
                    interpretation='Autograd connectivity and nonzero entries for this loss only; not effective capacity')


def construct(control, graph_path, vocabulary, seed):
    if control not in CONTROLS:
        raise ValueError('Unknown language core control: ' + str(control))
    torch.manual_seed(seed)
    graph = load_graph(graph_path)
    variant = 'no_recurrence' if control in ('no_lateral', 'no_temporal_state') else 'flm'
    return LanguageCoreControl(graph, CoreControlConfig(
        neurons=len(graph['body_ids']), pools=int(graph['pool'].max()) + 1,
        embedding=96, variant=variant, vocabulary=vocabulary, tied_readout=True,
        control=control))
