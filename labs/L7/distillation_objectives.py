#!/usr/bin/env python3
"""Masked FP64 CE / forward KL / reverse KL / generalized JSD references.

The beta endpoints follow TRL cd2c5287 explicitly: 0 -> forward KL,
1 -> reverse KL. Interior beta values select generalized JSD. Temperature
squared scaling is optional and is not enabled in that Trainer's loss.
"""
import torch
from torch import nn
from torch.nn import functional as F


def divergence(student, teacher, mask, beta=0., temperature=1., multiply_t2=False):
    if temperature <= 0 or not 0 <= beta <= 1:
        raise ValueError('positive temperature and beta in [0,1] required')
    ls = F.log_softmax(student / temperature, dim=-1)
    lt = F.log_softmax(teacher.detach() / temperature, dim=-1)
    ps, pt = ls.exp(), lt.exp()
    if beta == 0:
        per_token = (pt * (lt - ls)).sum(-1)
        gradient = (ps - pt) / temperature
    elif beta == 1:
        per_token = (ps * (ls - lt)).sum(-1)
        gradient = ps * ((ls - lt) - per_token[..., None]) / temperature
    else:
        b = student.new_tensor(beta)
        lm = torch.logsumexp(torch.stack([lt + b.log(), ls + torch.log1p(-b)]), dim=0)
        per_token = beta * (pt * (lt - lm)).sum(-1) + (1-beta) * (ps * (ls-lm)).sum(-1)
        log_ratio = ls - lm
        gradient = (1-beta) * ps * (log_ratio - (ps * log_ratio).sum(-1, keepdim=True)) / temperature
    denominator = mask.sum()
    if denominator == 0:
        return student.sum() * 0, torch.zeros_like(student)
    factor = temperature**2 if multiply_t2 else 1.
    loss = factor * (per_token * mask).sum() / denominator
    return loss, factor * gradient * mask[..., None] / denominator


def masked_ce(student, labels, mask, temperature=1.):
    logits = student / temperature
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none').reshape(labels.shape)
    count = mask.sum()
    if count == 0:
        return student.sum() * 0, torch.zeros_like(student)
    grad = (logits.softmax(-1) - F.one_hot(labels, logits.shape[-1])) / temperature
    return (per_token * mask).sum() / count, grad * mask[..., None] / count


def forward_kl_divergence(student, teacher, temperature=1.):
    return divergence(student, teacher, torch.ones_like(student[..., 0]), 0., temperature, True)[0]


def reverse_kl_divergence(student, teacher, temperature=1.):
    return divergence(student, teacher, torch.ones_like(student[..., 0]), 1., temperature, True)[0]


def feature_alignment_loss(student_hidden, teacher_hidden, projection):
    return F.mse_loss(projection(student_hidden), teacher_hidden.detach())


def verify_objectives():
    rows = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        student = torch.randn(2, 5, 3, dtype=torch.float64).transpose(1, 2).requires_grad_()
        teacher = torch.randn_like(student, requires_grad=True)
        mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
        labels = torch.randint(0, 5, (2, 3))
        for beta, name in [(0., 'forward_kl'), (.3, 'jsd'), (1., 'reverse_kl')]:
            loss, expected = divergence(student, teacher, mask, beta, temperature=4.)
            actual = torch.autograd.grad(loss, (student, teacher), retain_graph=True, allow_unused=True)
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=1e-12)
            assert actual[1] is None and torch.equal(actual[0][mask == 0], torch.zeros_like(actual[0][mask == 0]))
            rows.append({'seed': seed, 'objective': name, 'loss': loss.item(),
                         'max_gradient_error': float((actual[0]-expected).detach().abs().max()),
                         'teacher_has_gradient': actual[1] is not None, 'valid_positions': int(mask.sum())})
        loss, expected = masked_ce(student, labels, mask, 4.)
        actual = torch.autograd.grad(loss, student, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
        rows.append({'seed': seed, 'objective': 'ce', 'loss': loss.item(),
                     'max_gradient_error': float((actual-expected).detach().abs().max()), 'valid_positions': int(mask.sum())})
        empty, grad = divergence(student, teacher, torch.zeros_like(mask))
        assert empty.item() == 0 and torch.equal(grad, torch.zeros_like(grad))
    return rows


def verify_temperature_scaling():
    teacher = torch.tensor([[1., 2., -1.]], dtype=torch.float64)
    student = torch.tensor([[.5, -1., 1.]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, dtype=torch.float64)
    grads = {}
    for name, temp, square in [('T1', 1., False), ('T4', 4., False), ('T4_times_T2', 4., True)]:
        loss, _ = divergence(student, teacher, mask, temperature=temp, multiply_t2=square)
        grads[name] = torch.autograd.grad(loss, student, retain_graph=True)[0]
    torch.testing.assert_close(grads['T4_times_T2'], 16 * grads['T4'], rtol=0, atol=1e-12)
    shifted = (teacher + 7).requires_grad_()
    loss, _ = divergence(shifted, teacher, mask, temperature=100., multiply_t2=True)
    shift_grad = torch.autograd.grad(loss, shifted)[0]
    assert shift_grad.abs().max() < 1e-12
    return {'gradient_norms': {k: float(v.norm()) for k,v in grads.items()},
            'fixed_T_scaling_ratio': 16., 'constant_logit_shift_gradient': shift_grad.tolist(),
            'high_T_formula': '[(zs-mean(zs))-(zt-mean(zt))]/V after T^2 compensation',
            'note': 'The T=1 and T=4 compensated gradients are not generally equal.'}


def source_examples():
    torch.manual_seed(0)
    teacher_p = torch.tensor([.6, .3, .1], dtype=torch.float64)
    student_p = torch.tensor([.2, .5, .3], dtype=torch.float64)
    return {'synthetic_labels': [0, 1, 2, 0],
            'teacher_sampled_labels': torch.multinomial(teacher_p, 4, replacement=True).tolist(),
            'student_sampled_labels': torch.multinomial(student_p, 4, replacement=True).tolist(),
            'teacher_probabilities': teacher_p.tolist(), 'student_probabilities': student_p.tolist(),
            'scope': 'categorical input-source illustration; not text generation or task evaluation'}


def main():
    from _evidence import new_output, write_result
    torch.set_num_threads(1)
    out = new_output('Masked distillation objective and gradient contracts')
    torch.manual_seed(0)
    hs = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    projection = nn.Linear(3, 5).double()
    feature_alignment_loss(hs, ht, projection).backward()
    assert hs.grad is not None and ht.grad is None
    result = {'objective_parity': verify_objectives(), 'temperature': verify_temperature_scaling(),
              'input_sources': source_examples(), 'feature_only_reference': {'student_has_grad': True, 'teacher_has_grad': False,
              'not_an_eagle3_objective': True}}
    write_result(out, 'distillation_objectives_report.json', result,
                 {'device':'CPU', 'dtype':'float64', 'seeds':[0,1,2], 'logit_shape':[2,3,5],
                  'mask_valid_positions':3, 'temperature':4., 'beta':[0.,.3,1.],
                  'gradient_atol':1e-12, 'optimizer_updates':0, 'quality_measured':False}, [__file__])


if __name__ == '__main__':
    main()
