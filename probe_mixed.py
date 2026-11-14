import warnings, torch, botorch
from botorch.models import MixedSingleTaskGP
from botorch.utils.transforms import normalize_indices
from botorch.models.utils.assorted import check_min_max_scaling
print("botorch", botorch.__version__)
torch.manual_seed(0)
X = torch.cat([torch.rand(10, 2, dtype=torch.double),
               torch.randint(0, 3, (10, 1)).to(torch.double)], dim=-1)  # last col categorical in {0,1,2}
Y = torch.randn(10, 1, dtype=torch.double)
for cd in ([2], [-1]):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        m = MixedSingleTaskGP(X, Y, cat_dims=cd)
    msgs = [f"{w.category.__name__}: {w.message}" for w in ws if "unit cube" in str(w.message)]
    print(f"cat_dims={cd}: unit-cube warnings = {msgs}")
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    try:
        MixedSingleTaskGP(X, Y, cat_dims=[-1]) if False else None
        botorch.settings.validate_input_scaling  # noqa
        from botorch.exceptions import InputDataError
        X2 = X.clone()
        check_min_max_scaling(X2, raise_on_fail=True, ignore_dims=[-1])
        print("check_min_max_scaling(ignore_dims=[-1]) -> no error")
    except Exception as e:
        print("check_min_max_scaling(ignore_dims=[-1]) ->", type(e).__name__, e)
try:
    normalize_indices([-5], d=3)
except Exception as e:
    print("normalize_indices([-5], 3) ->", e)
