# %%
import numpy as np, xarray as xr, matplotlib.pyplot as plt


def _fit_func(x, p, A, B):
    return A * (p**x) + B


rng = np.random.default_rng(0)
t = np.arange(21)
da = xr.DataArray(
    np.stack(
        [
            _fit_func(t, 0.9, 0.8, 0),
            _fit_func(t, 0.7, 0.7, 0),
            _fit_func(t, 0.5, 0.6, 0),
        ]
    )
    + rng.normal(size=(3, t.size)) * 0.01,
    coords={"x": [0, 1, 2], "time": t},
)


ds_fit = da.curvefit("time", _fit_func, p0={"p": 0.99, "A": 1, "B": 0}, bounds={"p": (0, 1)})
params = ds_fit.curvefit_coefficients  # dims x,param

# Build fitted values DataArray
p = params.sel(param="p").values
A = params.sel(param="A").values
B = params.sel(param="B").values

fit_vals = np.stack([_fit_func(t, pi, Ai, Bi) for pi, Ai, Bi in zip(p, A, B)])
da_fit = xr.DataArray(fit_vals, coords={"x": da.x, "time": da.time}, dims=("x", "time"))

plt.figure()
for xi in da.x.values:
    plt.plot(da.time, da.sel(x=xi), "o", label=f"data x={xi}")
    plt.plot(da_fit.time, da_fit.sel(x=xi), "-", label=f"fit x={xi}")
plt.xlabel("time")
plt.ylabel("da")
plt.legend()
plt.title("Data and exponential fit")
plt.show()

params.to_pandas()
