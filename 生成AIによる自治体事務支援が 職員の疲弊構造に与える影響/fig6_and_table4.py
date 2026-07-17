# -*- coding: utf-8 -*-

def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from matplotlib import font_manager

    # ============================================================
    # Grouped (2-group) Relative-threshold Monte Carlo simulation
    #   - Two groups differ in lambda range (fatigue -> error sensitivity)
    #   - For each group, relative threshold theta_g is calibrated from:
    #       baseline None scenario's busy-window fatigue pooled (i,t), P90
    #   - Then evaluate None vs A+B+C within the SAME group using theta_g
    #   - Output:
    #       - Figure 6: group x {None, A+B+C} mean fatigue (busy window) with 95% CI
    #       - Table 4:  group-wise risk reduction (relative threshold) + mean fatigue
    # ============================================================

    # =========================
    # Japanese font setup - improved
    # =========================
    def setup_japanese_font():
        """日本語フォントを設定する（改善版）"""
        # まずIPAフォントをインストール
        try:
            import subprocess
            subprocess.run(['apt-get', 'update'], check=False, capture_output=True)
            subprocess.run(['apt-get', 'install', '-y', 'fonts-ipafont-gothic'], check=False, capture_output=True)
            print("IPAフォントをインストールしました")
        except Exception as e:
            print(f"フォントインストール試行: {e}")

        # フォントキャッシュをクリア
        try:
            font_manager._load_fontmanager(try_read_cache=False)
        except:
            pass

        font_paths = [
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
            "/usr/share/fonts/truetype/ipafont/ipagp.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
        ]

        font_found = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font_manager.fontManager.addfont(font_path)
                    fp = font_manager.FontProperties(fname=font_path)
                    plt.rcParams["font.family"] = fp.get_name()
                    plt.rcParams["axes.unicode_minus"] = False
                    print(f"✓ フォント設定完了: {fp.get_name()} ({font_path})")
                    font_found = True
                    break
                except Exception as e:
                    print(f"フォント読み込みエラー ({font_path}): {e}")
                    continue

        if not font_found:
            # フォールバック: sans-serifで日本語対応を試みる
            print("個別フォントが見つからなかったため、システムフォントを使用します")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'IPAPGothic', 'Noto Sans CJK JP', 'Takao PGothic']
            plt.rcParams["axes.unicode_minus"] = False

        return font_found

    # フォントセットアップを実行
    setup_japanese_font()

    # =========================
    # Reproducibility
    # =========================
    SEED = 42

    # =========================
    # Core settings
    # =========================
    N = 500          # agents
    T = 52           # weeks (365 for daily)
    MC = 1000        # Monte Carlo replications

    # Busy period window (index-based)
    BUSY_START = int(T * 0.25)
    BUSY_END   = int(T * 0.60)

    # Relative threshold definition (per group)
    THETA_PERCENTILE = 90  # baseline None busy-window P90

    # =========================
    # Group definitions
    #   - "Standard": lower lambda range
    #   - "HighSensitivity": higher lambda range (cognitive vulnerability proxy)
    # =========================
    GROUPS = {
        "標準群": {
            "lam_lo": 0.02,
            "lam_hi": 0.08,
        },
        "配慮群": {
            "lam_lo": 0.06,
            "lam_hi": 0.14,
        },
    }

    STABILITY_MARGIN = 0.95

    # =========================
    # Parameter sampling
    # =========================
    def sample_params(n, rng, lam_lo, lam_hi):
        rho   = rng.uniform(0.70, 0.90, n)
        alpha = rng.uniform(0.80, 1.20, n)
        beta  = rng.uniform(0.80, 1.20, n)
        gamma = rng.uniform(0.50, 1.00, n)
        kappa = rng.uniform(0.10, 0.20, n)
        lam   = rng.uniform(lam_lo, lam_hi, n)

        # stability-oriented clipping
        lam_cap = (STABILITY_MARGIN - rho) / np.maximum(beta, 1e-9)
        lam_cap = np.clip(lam_cap, 0.0, None)
        lam = np.minimum(lam, lam_cap)

        return rho, alpha, beta, gamma, kappa, lam

    # =========================
    # Workload / recovery processes
    # =========================
    def workload(t, rng):
        base = rng.normal(80, 15)
        if BUSY_START <= t <= BUSY_END:
            base += rng.normal(40, 10)
        return max(0.0, base)

    def recovery(t, rng):
        base = 40.0 if not (BUSY_START <= t <= BUSY_END) else 20.0
        base += rng.normal(0, 3)
        return max(0.0, base)

    # =========================
    # AI effects
    # =========================
    def apply_ai_params(rho_i, kappa_i, lam_i, mode):
        if "B" in mode:
            kappa_i *= 0.6
            lam_i   *= 0.6
        if "C" in mode:
            rho_i   *= 0.9
        return rho_i, kappa_i, lam_i

    # =========================
    # One simulation run
    # =========================
    def simulate_one(mode, rng, group_cfg):
        rho, alpha, beta, gamma, kappa, lam = sample_params(
            N, rng,
            lam_lo=group_cfg["lam_lo"],
            lam_hi=group_cfg["lam_hi"]
        )

        F = np.zeros((N, T), dtype=float)
        F[:, 0] = np.clip(rng.normal(50, 10, size=N), 0, None)

        for t in range(T - 1):
            W_base = workload(t, rng)
            R = recovery(t, rng)

            for i in range(N):
                W = W_base * 0.80 if "A" in mode else W_base
                rho_i, kappa_i, lam_i = apply_ai_params(rho[i], kappa[i], lam[i], mode)
                E = kappa_i * W + lam_i * F[i, t] + rng.normal(0, 2)
                F_next = (
                    rho_i * F[i, t]
                    + alpha[i] * W
                    + beta[i] * E
                    - gamma[i] * R
                    + rng.normal(0, 5)
                )
                F[i, t + 1] = max(0.0, F_next)

        return F

    # =========================
    # Metrics in busy window
    # =========================
    def metrics_busy(F, theta):
        busy = F[:, BUSY_START:BUSY_END+1]
        mean_f = float(np.mean(busy))
        risk   = float(np.mean(busy >= theta))
        p90    = float(np.percentile(busy, 90))
        return mean_f, risk, p90

    # =========================
    # Calibrate theta per group
    # =========================
    def calibrate_theta_for_group(group_cfg, seed_offset):
        """Relative threshold θ for each group:
        median of baseline (None) busy-window P90 across MC replications.
        """
        p_list = []
        for r in range(MC):
            sub_rng = np.random.default_rng(SEED * 10_000 + seed_offset + r)
            F0 = simulate_one("None", sub_rng, group_cfg)
            busy = F0[:, BUSY_START:BUSY_END+1].ravel()  # pool i,t
            p_list.append(np.percentile(busy, THETA_PERCENTILE))
        theta = float(np.median(p_list))
        return theta

    # =========================
    # Monte Carlo aggregation
    # =========================

    def run_mc(mode, theta, group_cfg, seed_offset):
        stats = np.zeros((MC, 3), dtype=float)
        for r in range(MC):
            sub_rng = np.random.default_rng(SEED * 10_000 + seed_offset + r)
            F = simulate_one(mode, sub_rng, group_cfg)
            stats[r, :] = metrics_busy(F, theta)

        df = pd.DataFrame(stats, columns=["mean", "risk", "p90"])
        out = {
            "mean": float(df["mean"].mean()),
            "mean_ci_lo": float(df["mean"].quantile(0.025)),
            "mean_ci_hi": float(df["mean"].quantile(0.975)),
            "risk": float(df["risk"].mean()),
            "risk_ci_lo": float(df["risk"].quantile(0.025)),
            "risk_ci_hi": float(df["risk"].quantile(0.975)),
            "p90": float(df["p90"].mean()),
            "p90_ci_lo": float(df["p90"].quantile(0.025)),
            "p90_ci_hi": float(df["p90"].quantile(0.975)),
        }
        return out

    # =========================
    # Helper: yerr
    # =========================
    def make_yerr(y, lo, hi):
        lower = np.clip(y - lo, 0, None)
        upper = np.clip(hi - y, 0, None)
        return np.vstack([lower, upper])

    # ============================================================
    # MAIN: compute Fig6 + Table4
    # ============================================================
    rows = []

    for gi, (gname, gcfg) in enumerate(GROUPS.items()):
        theta_g = calibrate_theta_for_group(gcfg, seed_offset=1_000_000 * gi )
        print(f"[{gname}] Calibrated θ (baseline None, busy P{THETA_PERCENTILE}) = {theta_g:.3f}")

        for sj, mode in enumerate(["None", "A+B+C"]):
            seed_offset = 1_000_000 * gi 
            out = run_mc(mode, theta_g, gcfg, seed_offset=seed_offset)
            rows.append({
                "group": gname,
                "mode": mode,
                "theta": theta_g,
                **out
            })

    df_g = pd.DataFrame(rows)
    df_g.to_csv("table_grouped_fig6.csv", index=False, encoding="utf-8-sig")
    print("Saved: table_grouped_fig6.csv")

    # Table 4
    table4 = []
    for gname in GROUPS.keys():
        d0 = df_g[(df_g["group"] == gname) & (df_g["mode"] == "None")].iloc[0]
        d1 = df_g[(df_g["group"] == gname) & (df_g["mode"] == "A+B+C")].iloc[0]
        rr = (d0["mean"] - d1["mean"]) / max(d0["mean"], 1e-12)

        table4.append({
            "群": gname,
            "相対閾値 θ（None群のbusy P90）": float(d0["theta"]),
            "AI支援なし 平均疲弊": float(d0["mean"]),
            "AI支援なし リスク": float(d0["risk"]),
            "A+B+C 平均疲弊": float(d1["mean"]),
            "A+B+C リスク": float(d1["risk"]),
            "平均疲弊 逓減率": float(rr),
        })

    df_table4 = pd.DataFrame(table4)
    df_table4.to_csv("table4_grouped.csv", index=False, encoding="utf-8-sig")
    print("Saved: table4_grouped.csv")

    # ============================================================
    # Figure 6
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = list(GROUPS.keys())
    # Display labels for the chart (keep internal group keys as-is)
    GROUP_LABEL = {"標準群": "Standard workers", "配慮群": "High-sensitivity workers"}
    group_labels = [GROUP_LABEL.get(g, g) for g in groups]
    x = np.arange(len(groups))
    width = 0.35

    means_none = []
    lo_none = []
    hi_none = []
    means_abc = []
    lo_abc = []
    hi_abc = []

    for gname in groups:
        d0 = df_g[(df_g["group"] == gname) & (df_g["mode"] == "None")].iloc[0]
        d1 = df_g[(df_g["group"] == gname) & (df_g["mode"] == "A+B+C")].iloc[0]

        means_none.append(d0["mean"])
        lo_none.append(d0["mean_ci_lo"])
        hi_none.append(d0["mean_ci_hi"])

        means_abc.append(d1["mean"])
        lo_abc.append(d1["mean_ci_lo"])
        hi_abc.append(d1["mean_ci_hi"])

    means_none = np.array(means_none, dtype=float)
    means_abc  = np.array(means_abc, dtype=float)

    yerr_none = make_yerr(means_none, np.array(lo_none), np.array(hi_none))
    yerr_abc  = make_yerr(means_abc,  np.array(lo_abc),  np.array(hi_abc))

    ax.bar(x - width/2, means_none, width, label="Without AI support (None)", color='#1f77b4')
    ax.errorbar(x - width/2, means_none, yerr=yerr_none, fmt="none", capsize=3, color='black')

    ax.bar(x + width/2, means_abc, width, label="With AI support (A+B+C)", color='#ff7f0e')
    ax.errorbar(x + width/2, means_abc, yerr=yerr_abc, fmt="none", capsize=3, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel("Worker group")
    ax.set_ylabel("Mean fatigue (busy-window average)")
    ax.set_title(
        f"Figure 6: Effects of combined generative AI support (A+B+C) on average fatigue by cognitive characteristics.\n"
        f"(weekly, busy window, relative θ = baseline P{THETA_PERCENTILE}, N={N}, MC={MC})"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig("fig6_group_effect_mean.png", dpi=300, bbox_inches='tight')
    print("Saved: fig6_group_effect_mean.png")
    plt.show()


if __name__ == '__main__':
    main()