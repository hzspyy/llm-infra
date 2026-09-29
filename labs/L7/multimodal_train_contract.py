#!/usr/bin/env python3
"""CPU contracts for VLM CE, masked conditional flow, and contrastive alignment.

Each seeded case performs one joint update, then one encoder-unfreeze update.
The model is a mathematical teaching model, not a Qwen/Cosmos implementation.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class FrozenVisionEncoder(nn.Module):
    def __init__(self, visual_dim=32):
        super().__init__()
        self.conv = nn.Linear(visual_dim, visual_dim)
        self.requires_grad_(False)

    def forward(self, images):
        # Frozen weights may still propagate a gradient to their inputs.
        return self.conv(images)


class VisionProjector(nn.Module):
    def __init__(self, visual_dim=32, llm_dim=16):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(visual_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))

    def forward(self, features):
        return self.proj(features)


class UnifiedMultimodalModel(nn.Module):
    def __init__(self, visual_dim=32, llm_dim=16, vocab_size=64):
        super().__init__()
        self.visual_encoder = FrozenVisionEncoder(visual_dim)
        self.connector = VisionProjector(visual_dim, llm_dim)
        self.shared_embedding = nn.Embedding(vocab_size, llm_dim)
        block = nn.TransformerEncoderLayer(llm_dim, 2, 32, dropout=0., batch_first=True)
        self.decoder = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
        self.lm_head = nn.Linear(llm_dim, vocab_size, bias=False)
        self.latent_projection = nn.Linear(visual_dim, llm_dim)
        self.time_projection = nn.Sequential(nn.Linear(1, llm_dim), nn.SiLU(), nn.Linear(llm_dim, llm_dim))
        self.flow_head = nn.Linear(llm_dim, visual_dim)

    def forward_vlm(self, images, text_tokens):
        visual = self.connector(self.visual_encoder(images))
        tokens = torch.cat([visual, self.shared_embedding(text_tokens)], dim=1)
        mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device).triu(1)
        return self.lm_head(self.decoder(tokens, mask=mask))

    def forward_flow(self, noisy_latent, text_tokens, t=None):
        if t is None:
            t = torch.zeros(noisy_latent.shape[0], dtype=noisy_latent.dtype, device=noisy_latent.device)
        condition = self.shared_embedding(text_tokens).mean(dim=1, keepdim=True)
        time = self.time_projection(t.reshape(-1, 1)).unsqueeze(1)
        return self.flow_head(torch.tanh(self.latent_projection(noisy_latent) + condition + time))

    def contrastive(self, images, text_tokens, temperature=.2):
        image = F.normalize(self.connector(self.visual_encoder(images)).mean(1), dim=-1)
        text = F.normalize(self.shared_embedding(text_tokens).mean(1), dim=-1)
        logits = image @ text.T / temperature
        labels = torch.arange(len(images), device=images.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def objectives(model, images, tokens, x0, x1, t, frame_mask):
    prefix = images.shape[1]
    labels = torch.cat([torch.full((len(tokens), prefix), -100, dtype=torch.long), tokens], dim=1)
    labels[-1, -1] = -100  # one padding/unsupervised target
    logits = model.forward_vlm(images, tokens)
    valid_tokens = int((labels[:, 1:] != -100).sum())
    vlm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction='sum') / valid_tokens
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    residual = model.forward_flow(xt, tokens, t) - (x1 - x0)
    coordinates = int(frame_mask.sum()) * x0.shape[-1]
    flow = (residual.square() * frame_mask[..., None]).sum() / coordinates
    contrastive = model.contrastive(images, tokens)
    return {'vlm_ce': vlm, 'flow_mse': flow, 'contrastive_ce': contrastive}, {
        'labels': labels.tolist(), 'valid_next_token_targets': valid_tokens,
        'frame_mask': frame_mask.tolist(), 'valid_flow_coordinates': coordinates,
        'contrastive_pairs': len(tokens), 'flow_times': t.tolist()}, xt


def verify_multimodal_training_step(seed=0):
    torch.manual_seed(seed)
    model = UnifiedMultimodalModel().double()
    images = torch.randn(2, 4, 32, dtype=torch.float64, requires_grad=True)
    tokens = torch.randint(0, 64, (2, 8))
    x0 = torch.randn(2, 4, 32, dtype=torch.float64)
    x1 = torch.randn_like(x0)
    t = torch.tensor([.2, .8], dtype=torch.float64)
    frame_mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]], dtype=torch.float64)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    connector = list(model.connector.parameters())
    connector_ids = {id(p) for p in connector}
    optimizer = torch.optim.AdamW([
        {'params': connector, 'lr': .01},
        {'params': [p for _, p in named if id(p) not in connector_ids], 'lr': .001}], weight_decay=0.)
    losses, counts, xt = objectives(model, images, tokens, x0, x1, t, frame_mask)
    gradients = {}
    for name, loss in losses.items():
        values = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        gradients[name] = {n: 0. if g is None else float(g.norm()) for (n, _), g in zip(named, values)}
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    total = losses['vlm_ce'] + .5 * losses['flow_mse'] + .1 * losses['contrastive_ce']
    total.backward()
    input_grad = float(images.grad.norm())
    assert input_grad > 0 and all(p.grad is None for p in model.visual_encoder.parameters())
    optimizer.step()
    deltas = {name: float((p.detach() - before[name]).abs().max()) for name, p in model.named_parameters()}
    assert all(v == 0 for n, v in deltas.items() if n.startswith('visual_encoder.'))
    assert deltas['lm_head.weight'] > 0 and deltas['flow_head.weight'] > 0
    with torch.no_grad():
        image_effect = float((model.forward_vlm(images + 1, tokens)[:, 4:] - model.forward_vlm(images, tokens)[:, 4:]).abs().max())
        latent_effect = float((model.forward_flow(xt + 1, tokens, t) - model.forward_flow(xt, tokens, t)).abs().max())
        time_effect = float((model.forward_flow(xt, tokens, t + .1) - model.forward_flow(xt, tokens, t)).abs().max())
    assert min(image_effect, latent_effect, time_effect) > 0
    encoder_params = list(model.visual_encoder.parameters())
    state_before = sum(p in optimizer.state for p in encoder_params)
    model.visual_encoder.requires_grad_(True)
    optimizer.add_param_group({'params': encoder_params, 'lr': .0001})
    optimizer.zero_grad(set_to_none=True)
    next_losses, _, _ = objectives(model, images, tokens, x0, x1, t, frame_mask)
    next_losses['vlm_ce'].backward()
    optimizer.step()
    state_after = sum(p in optimizer.state for p in encoder_params)
    assert state_before == 0 and state_after == len(encoder_params)
    return {'seed': seed, 'losses': {k: float(v.detach()) for k, v in losses.items()},
            'normalization': counts, 'gradient_norms_by_objective_and_parameter': gradients,
            'joint_update_max_abs_delta': deltas, 'frozen_encoder_input_grad_norm': input_grad,
            'input_dependence': {'image_to_text_logits': image_effect, 'latent_to_flow': latent_effect, 'time_to_flow': time_effect},
            'learning_rates': [.01, .001, .0001],
            'encoder_unfreeze': {'states_before': state_before, 'states_after': state_after,
                                'weight_delta_from_initial': float((model.visual_encoder.conv.weight.detach() - before['visual_encoder.conv.weight']).abs().max())},
            'updates': 2, 'quality_measured': False}


def verify_multimodal_failure_traps():
    import warnings
    model = UnifiedMultimodalModel().double()
    shared = model.shared_embedding.weight
    try:
        torch.optim.AdamW([{'params': [shared]}, {'params': [shared]}])
    except ValueError as error:
        duplicate = {'error_type': type(error).__name__, 'message': str(error)}
    else:
        raise AssertionError('Expected rejection of parameters repeated across groups')
    # A duplicate within one group is a different API case; record this version's behavior.
    repeated = nn.Parameter(torch.tensor(1., dtype=torch.float64))
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        try:
            repeated_opt = torch.optim.SGD([repeated, repeated], lr=.1)
            repeated.grad = torch.tensor(2., dtype=torch.float64)
            repeated_opt.step()
            within_group = {'parameter_after': repeated.item(), 'error': None}
        except ValueError as error:
            within_group = {'parameter_after': repeated.item(), 'error': str(error)}
        within_group['warnings'] = [str(w.message) for w in observed]
    model.visual_encoder.requires_grad_(True)
    x = torch.ones(2, 4, 32, dtype=torch.float64)
    cached = model.visual_encoder(x).detach()
    model.connector(cached).square().mean().backward()
    assert all(p.grad is None for p in model.visual_encoder.parameters())
    stale = {'encoder_gradients': [None if p.grad is None else float(p.grad.norm()) for p in model.visual_encoder.parameters()],
             'connector_has_gradient': any(p.grad is not None for p in model.connector.parameters())}
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    big = theta * .001 + 3.5
    small = theta * 10 + .015
    gradient = {'loss_values': [float(big.detach()), float(small.detach())],
                'gradients': [float(torch.autograd.grad(big, theta)[0]), float(torch.autograd.grad(small, theta)[0])]}
    return {'duplicate_param_in_optimizer': duplicate, 'duplicate_within_one_group': within_group,
            'stale_cache_with_unfrozen_encoder': stale,
            'loss_values_do_not_determine_gradients': gradient}


def main():
    from _evidence import new_output, write_result
    torch.set_num_threads(1)
    out = new_output('Multimodal objectives, input dependence, freezing and actual optimizer updates')
    result = {'steps': [verify_multimodal_training_step(seed) for seed in (0, 1, 2)],
              'failure_cases': verify_multimodal_failure_traps()}
    write_result(out, 'multimodal_train_report.json', result,
                 {'device': 'CPU', 'dtype': 'float64', 'seeds': [0, 1, 2], 'updates_per_seed': 2,
                  'shapes': {'images': [2, 4, 32], 'text': [2, 8], 'latent': [2, 4, 32]},
                  'targets': ['shifted masked CE', 'masked conditional flow MSE', 'symmetric contrastive CE'],
                  'quality_or_performance_measured': False}, [__file__])


if __name__ == '__main__':
    main()
