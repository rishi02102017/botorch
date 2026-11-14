import torch, botorch
from botorch.utils.multi_objective.pareto import is_non_dominated, _is_non_dominated_loop
print("botorch", botorch.__version__, "torch", torch.__version__)
nan = float("nan")

def show(label, f):
    try:
        print(label, "->", f())
    except Exception as e:
        print(label, "-> EXC", type(e).__name__, e)

# Case A: partial NaN at index 0 (docstring: any NaN -> False)
Y = torch.tensor([[nan, 1.0], [0.0, 0.0]])
show("A  partial NaN idx0, dedup=True ", lambda: is_non_dominated(Y))
# Case B: partial NaN at non-zero index, deduplicate=False
Y = torch.tensor([[0.0, 0.0], [nan, 1.0]])
show("B  partial NaN idx1, dedup=False", lambda: is_non_dominated(Y, deduplicate=False))
show("B' partial NaN idx1, dedup=True ", lambda: is_non_dominated(Y))
# Case C: loop path (n>1000): NaN point is kept AND wrongly dominates a valid point
Y = torch.zeros(1001, 2); Y[1:] = -1.0  # point 0 is the unique optimum
Y[0] = torch.tensor([0.0, 0.0]); Y[1] = torch.tensor([nan, 5.0])
Y[2] = torch.tensor([1.0, 1.0])  # true unique Pareto point (dominates everyone valid)
m = is_non_dominated(Y)
print("C  loop path n=1001: mask[:3] =", m[:3].tolist(), " total True =", int(m.sum()), " (expected [False, False, True], 1)")
m2 = is_non_dominated(Y[:3])
print("C' same 3 points, vectorized path mask =", m2.tolist())
# Case D: loop path, NaN point dominates a valid point that should be non-dominated
Y = torch.full((1001, 2), -10.0)
Y[0] = torch.tensor([nan, 5.0]); Y[1] = torch.tensor([1.0, 1.0])
m = is_non_dominated(Y)
print("D  loop path: mask[:2] =", m[:2].tolist(), " expected [False, True]")
m = _is_non_dominated_loop(Y[:2])
print("D' _is_non_dominated_loop on 2 pts:", m.tolist())
# minimize
Y = torch.tensor([[nan, 1.0], [2.0, 2.0]])
show("E  minimize, partial NaN idx0", lambda: is_non_dominated(Y, maximize=False))
# batched
Y = torch.tensor([[[nan, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, nan]]])
show("F  batched", lambda: is_non_dominated(Y))
