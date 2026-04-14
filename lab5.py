import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse

def display_stats_table(stats_data, distribution):
    print(f"\n" + "#"*70)
    print(f"[{distribution.upper()}] - СТАТИСТИКА КОЭФФИЦИЕНТОВ")
    print("#"*70)
    
    for coef_name in stats_data.keys():
        print(f"\n>>> Метрика: {coef_name} <<<")
        
        n_keys = sorted(stats_data[coef_name]["Среднее"][distribution].keys())
        first_n = n_keys[0]
        rho_keys = sorted(stats_data[coef_name]["Среднее"][distribution][first_n].keys())
        
        header = f"{'n':<6}"
        for r in rho_keys:
            header += f"|  rho = {r} ( E / V )  ".center(25)
        
        print(header)
        print("-" * len(header))

        for n in n_keys:
            row = f"{n:<6}"
            for r in rho_keys:
                mean_val = stats_data[coef_name]["Среднее"][distribution][n][r]
                var_val = stats_data[coef_name]["Дисперсия"][distribution][n][r]
                block = f"| {mean_val:7.3f} / {var_val:6.3f} "
                row += block.center(25)
            print(row)

def quadrant_corr_coef(x, y):
    med_x, med_y = np.median(x), np.median(y)
    dx = x - med_x
    dy = y - med_y
    
    q1 = np.sum((dx > 0) & (dy > 0))
    q2 = np.sum((dx < 0) & (dy > 0))
    q3 = np.sum((dx < 0) & (dy < 0))
    q4 = np.sum((dx > 0) & (dy < 0))

    return (q1 + q3 - q2 - q4) / len(x)

def draw_covariance_ellipses(n_size, min_v, max_v, rhos, title, sigma_level=3.0):
    fig, axes = plt.subplots(1, len(rhos), figsize=(5.5 * len(rhos), 5))
    
    if len(rhos) == 1:
        axes = [axes]
        
    palette = {0: 'teal', 0.5: 'darkorange', 0.9: 'crimson', 'mix': 'indigo'}
    
    for ax, rho_val in zip(axes, rhos):
        x = (min_v[n_size][rho_val]['x'] + max_v[n_size][rho_val]['x']) / 2
        y = (min_v[n_size][rho_val]['y'] + max_v[n_size][rho_val]['y']) / 2
        
        c_color = palette.get(rho_val, 'black')
        
        ax.scatter(x, y, s=20, alpha=0.4, color=c_color)
        
        cov_mat = np.cov(x, y)
        evals, evecs = np.linalg.eigh(cov_mat)
        
        sort_indices = evals.argsort()[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]
        
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        w, h = 2 * sigma_level * np.sqrt(evals)
        cx, cy = np.mean(x), np.mean(y)
        
        ellipse_patch = Ellipse(
            xy=(cx, cy),
            width=w,
            height=h,
            angle=angle,
            facecolor='none',   
            edgecolor=c_color,    
            linestyle='--',      
            linewidth=2.0,
            label=f'Эллипс ({sigma_level}$\sigma$)'
        )
        ax.add_patch(ellipse_patch)
        
        ax.plot(cx, cy, marker='D', color=c_color, markersize=6)

        ax.axhline(0, color='gray', lw=1.0, alpha=0.4, linestyle='-')
        ax.axvline(0, color='gray', lw=1.0, alpha=0.4, linestyle='-')
        
        ax.set_aspect('equal', 'datalim')
        
        label_dist = "Смешанное" if rho_val == "mix" else f"ρ = {rho_val}"
        ax.set_title(label_dist, fontweight='bold')
        
        ax.set_xlabel("Ось X")
        ax.set_ylabel("Ось Y")
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right') 
        
    fig.suptitle(f"{title} (Объем выборки n={n_size})", fontsize=15, y=1.03)
    
    plt.tight_layout()
    plt.show()
    
def main():
    rng = np.random.default_rng(42)
    
    distributions = ["Нормальное", "Смешанное"]
    sizes_n = [20, 60, 100]
    rho_list = [0, 0.5, 0.9]
    n_iters = 1000

    correlations = ["Пирсон", "Спирмен", "Квадрантный"]
    
    results_dict = {
        c: {d: {n: {r: [] for r in rho_list} for n in sizes_n} for d in distributions}
        for c in correlations
    }

    low_bnd = {
        n: {r: {'x': np.full(n, np.inf), 'y': np.full(n, np.inf)} for r in rho_list + ["mix"]} 
        for n in sizes_n
    }
    upp_bnd = {
        n: {r: {'x': np.full(n, -np.inf), 'y': np.full(n, -np.inf)} for r in rho_list + ["mix"]} 
        for n in sizes_n
    }

    for _ in range(n_iters):
        for n in sizes_n:
            for r in rho_list:
                cov = [[1, r], [r, 1]]
                smpl = rng.multivariate_normal([0, 0], cov, size=n)
                x, y = smpl[:, 0], smpl[:, 1]
                
                low_bnd[n][r]['x'] = np.minimum(low_bnd[n][r]['x'], x)
                upp_bnd[n][r]['x'] = np.maximum(upp_bnd[n][r]['x'], x)
                low_bnd[n][r]['y'] = np.minimum(low_bnd[n][r]['y'], y)
                upp_bnd[n][r]['y'] = np.maximum(upp_bnd[n][r]['y'], y)

                results_dict["Пирсон"]["Нормальное"][n][r].append(stats.pearsonr(x, y)[0])
                results_dict["Спирмен"]["Нормальное"][n][r].append(stats.spearmanr(x, y)[0])
                results_dict["Квадрантный"]["Нормальное"][n][r].append(quadrant_corr_coef(x, y))

            cov1, cov2 = [[1, 0.9], [0.9, 1]], [[10, -9], [-9, 10]]
            
            s1 = rng.multivariate_normal([0, 0], cov1, size=n)
            s2 = rng.multivariate_normal([0, 0], cov2, size=n)

            mix_smpl = s1 * 0.9 + s2 * 0.1
            mx, my = mix_smpl[:, 0], mix_smpl[:, 1]

            low_bnd[n]["mix"]['x'] = np.minimum(low_bnd[n]["mix"]['x'], mx)
            upp_bnd[n]["mix"]['x'] = np.maximum(upp_bnd[n]["mix"]['x'], mx)
            low_bnd[n]["mix"]['y'] = np.minimum(low_bnd[n]["mix"]['y'], my)
            upp_bnd[n]["mix"]['y'] = np.maximum(upp_bnd[n]["mix"]['y'], my)

            results_dict["Пирсон"]["Смешанное"][n][0].append(stats.pearsonr(mx, my)[0])
            results_dict["Спирмен"]["Смешанное"][n][0].append(stats.spearmanr(mx, my)[0])
            results_dict["Квадрантный"]["Смешанное"][n][0].append(quadrant_corr_coef(mx, my))

    summary_stats = {c: {"Среднее": {}, "Дисперсия": {}} for c in correlations}

    for c in correlations:
        for d in distributions:
            summary_stats[c]["Среднее"][d] = {n: {} for n in sizes_n}
            summary_stats[c]["Дисперсия"][d] = {n: {} for n in sizes_n}
            
            for n in sizes_n:
                active_r = rho_list if d == "Нормальное" else [0]
                
                for r in active_r:
                    values = results_dict[c][d][n][r]
                    summary_stats[c]["Среднее"][d][n][r] = np.mean(values)
                    summary_stats[c]["Дисперсия"][d][n][r] = np.var(values)

    display_stats_table(summary_stats, "Нормальное")
    display_stats_table(summary_stats, "Смешанное")

    for n in sizes_n:   
        draw_covariance_ellipses(
            n_size=n, 
            min_v=low_bnd, 
            max_v=upp_bnd, 
            rhos=[0, 0.5, 0.9], 
            title="Нормальное распределение", 
            sigma_level=3.0 
        )
        
        draw_covariance_ellipses(
            n_size=n, 
            min_v=low_bnd, 
            max_v=upp_bnd, 
            rhos=["mix"], 
            title="Смешанное распределение", 
            sigma_level=3.0 
        )

if __name__ == "__main__":
    main()