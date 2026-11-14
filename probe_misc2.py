import torch, botorch, traceback
print("botorch", botorch.__version__)
def show(label, f):
    try:
        print(label, "->", f())
    except Exception as e:
        print(label, "-> EXC", type(e).__name__, str(e)[:300])

from botorch.utils.probability import TruncatedMultivariateNormal
tk = dict(dtype=torch.float64)
d = 2
loc = torch.zeros(d, **tk)
cov = torch.eye(d, **tk)
bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], **tk)
tmvn = TruncatedMultivariateNormal(loc=loc, covariance_matrix=cov, bounds=bounds)
inb = torch.tensor([[0.1, 0.2], [0.3, -0.2]], **tk)
oob = torch.tensor([[5.0, 0.0], [0.0, 5.0]], **tk)
mix = torch.tensor([[0.1, 0.2], [0.0, 5.0]], **tk)
for nm, v in [("in-bounds", inb), ("all-oob", oob), ("mixed", mix)]:
    show(f"TMVN unbatched float64 log_prob {nm}", lambda v=v: (lambda r: (r.tolist(), r.dtype, tuple(r.shape)))(tmvn.log_prob(v)))
# batched distribution, single value
locb = torch.zeros(3, d, **tk)
covb = torch.eye(d, **tk).expand(3, d, d)
bb = bounds.expand(3, d, 2)
tb = TruncatedMultivariateNormal(loc=locb, covariance_matrix=covb, bounds=bb)
show("TMVN batch(3) value in-bounds (d,)", lambda: (lambda r: (r.dtype, tuple(r.shape)))(tb.log_prob(torch.tensor([0.1, 0.2], **tk))))
show("TMVN batch(3) value oob (d,)", lambda: (lambda r: (r.dtype, tuple(r.shape)))(tb.log_prob(torch.tensor([5.0, 0.2], **tk))))
show("TMVN equal bounds accepted?", lambda: TruncatedMultivariateNormal(loc=loc, covariance_matrix=cov, bounds=torch.tensor([[0.0, 0.0], [-1.0, 1.0]], **tk)).bounds.shape)

from botorch.sampling.qmc import NormalQMCEngine
e1 = NormalQMCEngine(3, seed=0); a = e1.draw(4, dtype=torch.float64)
e2 = NormalQMCEngine(3, seed=0); out = torch.empty(4, 3, dtype=torch.float64); e2.draw(4, out=out)
show("NormalQMCEngine out=float64 matches dtype=float64 draw", lambda: (torch.equal(a, out), (a - out).abs().max().item()))

from botorch.utils.feasible_volume import estimate_feasible_volume
from botorch.models.deterministic import GenericDeterministicModel
m = GenericDeterministicModel(lambda X: X.sum(-1, keepdim=True), num_outputs=1)
ic = [(torch.tensor([0]), torch.tensor([1.0]), 2.0)]  # x0 >= 2 infeasible in [0,1]
show("estimate_feasible_volume infeasible linear", lambda: estimate_feasible_volume(torch.tensor([[0.0, 0.0], [1.0, 1.0]]), m, [lambda Y: Y[..., 0] - 10], inequality_constraints=ic, nsample_feature=100, nsample_outcome=10, seed=0))

gm = GenericDeterministicModel(lambda X: torch.stack([X.sum(-1), X.prod(-1)], -1).expand(2, *X.shape[:-1], 2) if X.dim() < 4 else None, num_outputs=2, batch_shape=torch.Size([2]))
show("GenericDeterministicModel.subset_output batch_shape", lambda: (gm.batch_shape, gm.subset_output([0]).batch_shape))

from botorch.utils.transforms import normalize, unnormalize
b = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
show("normalize/unnormalize roundtrip const bound", lambda: unnormalize(normalize(torch.tensor([[1.0, 1.0]]), b), b))
from botorch.utils.sampling import draw_sobol_samples, sample_simplex, sample_hypersphere, manual_seed
show("sample_simplex d=1", lambda: sample_simplex(1, n=2))
show("sample_simplex float64", lambda: sample_simplex(3, n=2, dtype=torch.float64).dtype)
show("sample_hypersphere d=1", lambda: sample_hypersphere(1, n=2))
show("draw_sobol_samples batch bounds", lambda: draw_sobol_samples(torch.tensor([[[0.0],[1.0]],[[5.0],[6.0]]]), n=2, q=1).shape)
from botorch.utils.multi_objective.hypervolume import infer_reference_point
show("infer_reference_point 1 point", lambda: infer_reference_point(torch.tensor([[1.0, 2.0]])))
show("infer_reference_point float64 + max_ref float32", lambda: infer_reference_point(torch.tensor([[1.0, 2.0],[2.0,1.0]], dtype=torch.float64), max_ref_point=torch.tensor([0.5, 0.5])).dtype)
