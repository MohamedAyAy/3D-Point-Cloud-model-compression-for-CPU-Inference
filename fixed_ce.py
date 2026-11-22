"""fixed_ce.py - Cross-entropy et KL corrigees (Barron, Gnanapragasam, Tschannen, arXiv 2610.31642), version PyTorch.
  fixed_ce(z, y)            : CE a cible dure, corrige Softmax Collapse   (z: [M,C], y: [M] int64)
  fixed_kl(zs, zt, T)       : KL(softmax(zt/T) || softmax(zs/T)), corrige Softmax Cancellation (valeur = KL, mean sur M)
Test : python fixed_ce.py
"""
import torch, torch.nn.functional as F


def fixed_ce(z, y, reduction="mean"):
    onehot = F.one_hot(y, z.shape[-1]).bool()
    zt = z.gather(1, y[:, None])
    diff = (z - zt).masked_fill(onehot, float("-inf"))     # slot cible remplace par une constante : zt n'apparait que via z_j - z_t
    m = -torch.logsumexp(diff, dim=-1)                     # marge
    loss = F.softplus(-m)                                  # log(1 + e^-m), exact meme quand 1 + e^-m s'arrondit a 1
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


def fixed_kl(zs, zt, T=1.0, reduction="mean"):
    a, b = zs / T, zt / T
    with torch.no_grad():
        p = torch.softmax(b, -1); log_q = torch.log_softmax(a, -1); q = log_q.exp()
        delta = a - b
        d = delta - (p * delta).sum(-1, keepdim=True)                    # centre sous p
        e = torch.where(p > 0, p * torch.expm1(d.clamp(max=80.0)), torch.zeros_like(p))
        s = d - torch.log1p(e.sum(-1, keepdim=True))                     # log(q/p)
        u = torch.where(s.abs() < 1, p * torch.expm1(s), q - p)          # q - p sans soustraire deux probabilites
        g = u - q * u.sum(-1, keepdim=True)                              # somme des gradients = 0
        fwd = (p * (torch.log(p.clamp_min(1e-45)) - log_q)).sum(-1)     # valeur de la KL
    # gradient impose : d loss / d a = g
    loss = fwd + (g * (a - a.detach())).sum(-1)
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


if __name__ == "__main__":
    torch.manual_seed(0)
    # 1) Softmax Collapse : z = [9,-9,-9], cible 0
    z = torch.tensor([[9., -9., -9.]], requires_grad=True); y = torch.tensor([0])
    F.cross_entropy(z, y).backward(); gs = z.grad.clone()
    z.grad = None; fixed_ce(z, y).backward(); gf = z.grad.clone()
    print("CE standard  grad :", gs.numpy(), " somme =", gs.sum().item())
    print("CE corrigee  grad :", gf.numpy(), " somme =", gf.sum().item())
    ref = torch.softmax(torch.tensor([[9., -9., -9.]]).double(), -1); ref[0, 0] -= 1
    print("exact (float64)   :", ref.numpy())
    # 2) Softmax Cancellation : zS tres proche de zT
    zT = torch.tensor([[2., 1., 0.]]); zS = (zT + torch.tensor([[-1e-7, 1e-7, 0.]])).requires_grad_(True)
    F.kl_div(F.log_softmax(zS, -1), F.softmax(zT, -1), reduction="batchmean").backward(); gs = zS.grad.clone()
    zS.grad = None; fixed_kl(zS, zT, 1.0).backward(); gf = zS.grad.clone()
    ref = torch.softmax(zS.detach().double(), -1) - torch.softmax(zT.double(), -1)
    print("KL standard  grad :", gs.numpy()); print("KL corrigee  grad :", gf.numpy()); print("exact (float64)   :", ref.numpy())
    # 3) Coherence sur donnees aleatoires normales : memes valeurs et memes gradients que le standard
    zz = (torch.randn(1000, 17) * 3).requires_grad_(True); yy = torch.randint(0, 17, (1000,)); tt = torch.randn(1000, 17) * 3
    l1 = F.cross_entropy(zz, yy); g1, = torch.autograd.grad(l1, zz); l2 = fixed_ce(zz, yy); g2, = torch.autograd.grad(l2, zz)
    print(f"CE  : |dL| = {abs(l1 - l2).item():.2e}   max|dg| = {(g1 - g2).abs().max().item():.2e}")
    T = 4.0
    k1 = F.kl_div(F.log_softmax(zz / T, -1), F.softmax(tt / T, -1), reduction="batchmean"); g1, = torch.autograd.grad(k1, zz)
    k2 = fixed_kl(zz, tt, T); g2, = torch.autograd.grad(k2, zz)
    print(f"KL  : |dL| = {abs(k1 - k2).item():.2e}   max|dg| = {(g2 - g1).abs().max().item():.2e}")