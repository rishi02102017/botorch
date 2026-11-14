import torch, warnings
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.models.transforms.outcome import Standardize, Log
from botorch.models.transforms.input import Normalize
from botorch.utils.sampling import draw_sobol_samples
def show(label, f):
    try:
        print(label, "->", f())
    except Exception as e:
        print(label, "-> EXC", type(e).__name__, e)
show("standardize 1pt", lambda: standardize(torch.tensor([[1.0, 2.0]])))
show("Standardize 1pt", lambda: Standardize(m=2)(torch.tensor([[1.0, 2.0]])))
show("Standardize 1pt Yvar", lambda: Standardize(m=1)(torch.tensor([[1.0]]), torch.tensor([[0.1]])))
show("Normalize const col", lambda: Normalize(d=2)(torch.tensor([[1.0, 2.0],[1.0, 3.0]])))
show("sobol eq bounds", lambda: draw_sobol_samples(torch.tensor([[0.0, 1.0],[0.0, 2.0]]), n=2, q=1))
show("sobol float32 bounds dtype", lambda: draw_sobol_samples(torch.tensor([[0.0],[1.0]]), n=2, q=1).dtype)
