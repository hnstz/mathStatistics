import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

sample_sizes = [10, 100, 1000]

distributions = [
    {"name": "Normal N(0,1)", "func": stats.norm, "params": {"loc": 0, "scale": 1}, "x_range": np.linspace(-4, 4, 1000)},
    {"name": "Cauchy C(0,1)", "func": stats.cauchy, "params": {"loc": 0, "scale": 1}, "x_range": np.linspace(-10, 10, 1000)},
    {"name": "Laplace L(0, 1/sqrt(2))", "func": stats.laplace, "params": {"loc": 0, "scale": 1/np.sqrt(2)}, "x_range": np.linspace(-5, 5, 1000)},
    {"name": "Poisson P(10)", "func": stats.poisson, "params": {"mu": 10}, "x_range": np.arange(0, 21)},
    {"name": "Uniform U(-sqrt(3), sqrt(3))", "func": stats.uniform, "params": {"loc": -np.sqrt(3), "scale": 2*np.sqrt(3)}, "x_range": np.linspace(-2, 2, 1000)}
]

for dist in distributions:
    plt.figure(figsize=(12, 4))
    
    for i, n in enumerate(sample_sizes):
        plt.subplot(1, 3, i+1)
        
        
        if dist["name"] == "Cauchy C(0,1)":
            sample = dist["func"].rvs(size=n, **dist["params"])
            plt.hist(sample, bins=np.linspace(-10, 10, 30), density=True, alpha=0.7, edgecolor='black')
            theoretical = dist["func"].pdf(dist["x_range"], **dist["params"])
            plt.plot(dist["x_range"], theoretical, 'r-', linewidth=2)
            plt.xlim(-10, 10)

        elif dist["name"] == "Poisson P(10)":
            sample = dist["func"].rvs(size=n, **dist["params"])
            plt.hist(sample, bins=np.arange(-0.5, 21, 1), density=True, alpha=0.7, edgecolor='black')
            theoretical = dist["func"].pmf(dist["x_range"], **dist["params"])
            plt.scatter(dist["x_range"], theoretical, color='red', s=20, zorder=3)
            plt.xlim(-1, 21)
        else:
            sample = dist["func"].rvs(size=n, **dist["params"])
            plt.hist(sample, bins=20, density=True, alpha=0.7, edgecolor='black')
            theoretical = dist["func"].pdf(dist["x_range"], **dist["params"])
            plt.plot(dist["x_range"], theoretical, 'r-', linewidth=2)
            plt.xlim(dist["x_range"][0], dist["x_range"][-1])
            
        plt.title(f"n={n}")
        plt.xlabel("x")
        plt.ylabel("Плотность")
    
    plt.suptitle(dist["name"])
    plt.tight_layout()
    plt.show()