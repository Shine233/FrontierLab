#!/usr/bin/env python3
"""Generate geometry-concept figures for the 3D-reconstruction prerequisites doc.

These concepts (pinhole camera, epipolar geometry, triangulation, depth-vs-pointmap)
have no arXiv source figure, so we draw them ourselves. Run:
    python3 scripts/gen_prereq_figs.py
Outputs into assets/learning/3d-reconstruction-prereq/.
ponytail: pure matplotlib, no seaborn/styling deps; deterministic (no RNG).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CJK font (Noto Sans CJK JP carries the Chinese glyphs on this box)
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(
    os.path.dirname(__file__), "..", "assets", "learning", "3d-reconstruction-prereq"
)
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#38a169", "#e53e3e", "#718096"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(p))


# ---------------------------------------------------------------- pinhole camera
def pinhole():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # optical center
    C = np.array([0, 0])
    ax.plot(*C, "o", color=RED, ms=9, zorder=5)
    ax.annotate("光心 C\n(相机原点)", C, textcoords="offset points", xytext=(-6, -34),
                ha="center", color=RED, fontsize=10)
    # optical axis
    ax.annotate("", xy=(5.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(5.25, 0, "光轴 Z", va="center", color=GREY, fontsize=10)
    # image plane at focal length f
    f = 2.0
    ax.plot([f, f], [-1.5, 1.5], color=BLUE, lw=2)
    ax.text(f, 1.65, "像平面 (Z=f)", ha="center", color=BLUE, fontsize=10)
    ax.annotate("", xy=(f, -1.9), xytext=(0, -1.9),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text(f / 2, -2.1, "焦距 f", ha="center", color=GREY, fontsize=9)
    # 3D point
    P = np.array([4.5, 2.4])
    ax.plot(*P, "o", color=GREEN, ms=9, zorder=5)
    ax.text(P[0] + 0.15, P[1], "3D 点 P=(X,Y,Z)", va="center", color=GREEN, fontsize=10)
    # projection ray C->P, and image point p where it crosses image plane
    t = f / P[0]
    p = C + t * (P - C)
    ax.plot([C[0], P[0]], [C[1], P[1]], "--", color=GREEN, lw=1.2)
    ax.plot(*p, "o", color=ORANGE, ms=8, zorder=6)
    ax.text(p[0] + 0.1, p[1] + 0.15, "投影点 p=(u,v)", color=ORANGE, fontsize=10)
    # depth bracket
    ax.annotate("", xy=(P[0], -0.15), xytext=(0, -0.15),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.8))
    ax.text(P[0] / 2, -0.45, "深度 Z", ha="center", color=GREEN, fontsize=9)
    ax.set_xlim(-1, 6.5)
    ax.set_ylim(-2.6, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("针孔相机模型：3D 点沿视线投影到像平面", fontsize=12)
    save(fig, "pinhole.png")


# ------------------------------------------------------------- epipolar geometry
def epipolar():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    C1 = np.array([0, 0]); C2 = np.array([4.5, 0])
    for C, name, dx in [(C1, "相机1 C1", -0.1), (C2, "相机2 C2", 0.1)]:
        ax.plot(*C, "o", color=RED, ms=9, zorder=5)
        ax.text(C[0] + dx, C[1] - 0.4, name, ha="center", color=RED, fontsize=10)
    # baseline
    ax.plot([C1[0], C2[0]], [0, 0], color=GREY, lw=1.5)
    ax.text(2.25, -0.32, "基线 (baseline)", ha="center", color=GREY, fontsize=9)
    # 3D point
    P = np.array([2.6, 3.1])
    ax.plot(*P, "o", color=GREEN, ms=9, zorder=5)
    ax.text(P[0] + 0.12, P[1], "3D 点 P", color=GREEN, fontsize=10)
    # rays
    for C in (C1, C2):
        d = (P - C); d = d / np.linalg.norm(d)
        end = C + d * 4.2
        ax.plot([C[0], end[0]], [C[1], end[1]], "--", color=GREEN, lw=1.1)
    # image planes (short segments) + projected points
    for C, sgn in [(C1, 1), (C2, -1)]:
        base = C + np.array([sgn * 1.15, 1.15])
        seg = np.array([-0.6, 0.6])
        xs = base[0] + seg * 0.5
        ys = base[1] + seg * 0.9
        ax.plot(xs, ys, color=BLUE, lw=2)
        # projected point where ray C->P crosses (approx midpoint)
        d = (P - C); d = d / np.linalg.norm(d)
        pp = C + d * 1.6
        ax.plot(*pp, "o", color=ORANGE, ms=7, zorder=6)
    ax.text(1.0, 1.75, "投影 p1", color=ORANGE, fontsize=9)
    ax.text(3.35, 1.75, "投影 p2", color=ORANGE, fontsize=9)
    # epipolar line annotation
    ax.annotate("p1 对应的极线\n(p2 必在此线上)", xy=(3.5, 1.5), xytext=(4.7, 2.6),
                color=BLUE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    ax.set_xlim(-1, 7.2)
    ax.set_ylim(-0.9, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("对极几何：一张图的点，对应另一张图中的一条极线", fontsize=12)
    save(fig, "epipolar.png")


# ---------------------------------------------------------------- triangulation
def triangulation():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    C1 = np.array([0, 0]); C2 = np.array([5, 0])
    P = np.array([2.5, 3.0])
    for C, name in [(C1, "C1"), (C2, "C2")]:
        ax.plot(*C, "o", color=RED, ms=9, zorder=5)
        ax.text(C[0], C[1] - 0.4, name, ha="center", color=RED, fontsize=11)
    # two rays intersecting at P
    for C in (C1, C2):
        d = (P - C); d = d / np.linalg.norm(d)
        end = C + d * 4.5
        ax.plot([C[0], end[0]], [C[1], end[1]], "-", color=BLUE, lw=1.4)
        ax.annotate("", xy=end, xytext=C,
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.1, alpha=0))
    ax.plot(*P, "o", color=GREEN, ms=11, zorder=6)
    ax.text(P[0] + 0.15, P[1] + 0.1, "交点 = 重建的 3D 点 P", color=GREEN, fontsize=10)
    ax.text(1.0, 1.6, "视线1", color=BLUE, fontsize=9, rotation=50)
    ax.text(4.0, 1.6, "视线2", color=BLUE, fontsize=9, rotation=-50)
    ax.set_xlim(-0.8, 6.5)
    ax.set_ylim(-0.9, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("三角化：两条已知视线的交点恢复 3D 坐标", fontsize=12)
    save(fig, "triangulation.png")


# ------------------------------------------------ depth map vs point map
def depth_vs_pointmap():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    # left: depth map = per-pixel scalar
    ax = axes[0]
    grid = np.array([[3, 3, 4, 5],
                     [2, 3, 4, 5],
                     [2, 2, 3, 4],
                     [1, 2, 3, 4]], dtype=float)
    im = ax.imshow(grid, cmap="viridis")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    color="white", fontsize=10)
    ax.set_title("深度图 Depth Map\n每像素 1 个标量 Z", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Z (距离)")
    # right: point map = per-pixel 3D coord
    ax = axes[1]
    ax.set_title("点图 Point Map\n每像素 1 个三维坐标 (X,Y,Z)", fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, 3 - i, "(X,Y,Z)", ha="center", va="center", fontsize=7.5,
                    color=BLUE,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#ebf4ff", ec=BLUE, lw=0.7))
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(-0.6, 3.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("同一张图像的两种几何表示：深度图只给距离，点图直接给 3D 坐标",
                 fontsize=12, y=1.02)
    save(fig, "depth-vs-pointmap.png")


# ------------------------------------------------ self- vs cross-attention
def self_vs_cross_attention():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    def draw_seq(ax, x, labels, color, title):
        ys = np.linspace(2.6, 0.4, len(labels))
        pts = []
        for y, lab in zip(ys, labels):
            ax.add_patch(plt.Rectangle((x - 0.42, y - 0.16), 0.84, 0.32,
                         fc=color, ec="black", lw=0.8, zorder=3))
            ax.text(x, y, lab, ha="center", va="center", fontsize=9, zorder=4)
            pts.append((x, y))
        ax.text(x, 3.0, title, ha="center", fontsize=10, weight="bold")
        return pts

    # ---- self-attention: Q,K,V all from the SAME sequence ----
    ax = axes[0]
    src = draw_seq(ax, 0.8, ["tok1", "tok2", "tok3"], "#bee3f8", "同一组 token")
    # highlight tok2 as the query, attending to all (incl. itself)
    qx, qy = src[1]
    for (kx, ky) in src:
        ax.annotate("", xy=(kx + 0.42, ky), xytext=(qx, qy),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.3,
                                    connectionstyle="arc3,rad=0.28"), zorder=2)
    ax.text(qx, qy - 0.9, "Q 来自 tok2", ha="center", color=RED, fontsize=9)
    ax.text(1.9, 1.5, "K,V 来自\n同一组", ha="center", color=BLUE, fontsize=9)
    ax.set_title("自注意力 Self-Attention\nQ, K, V 同源（token 看自己人）", fontsize=11)
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 3.3); ax.axis("off")

    # ---- cross-attention: Q from seq A, K/V from seq B ----
    ax = axes[1]
    a = draw_seq(ax, 0.7, ["A1", "A2"], "#fed7aa", "序列 A (Query)")
    b = draw_seq(ax, 2.3, ["B1", "B2", "B3"], "#c6f6d5", "序列 B (Key/Value)")
    for (qx, qy) in a:
        for (kx, ky) in b:
            ax.annotate("", xy=(kx - 0.42, ky), xytext=(qx + 0.42, qy),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8,
                                        alpha=0.7), zorder=2)
    ax.text(1.5, 0.05, "A 的 query 去看 B 的 key/value", ha="center",
            color=ORANGE, fontsize=9)
    ax.set_title("交叉注意力 Cross-Attention\nQ 来自 A，K/V 来自 B（跨序列对齐）", fontsize=11)
    ax.set_xlim(0, 3.2); ax.set_ylim(-0.2, 3.3); ax.axis("off")

    fig.suptitle("自注意力 vs 交叉注意力：区别只在 Q 与 K/V 是否同源",
                 fontsize=12.5, y=1.02)
    save(fig, "self-vs-cross-attn.png")


if __name__ == "__main__":
    pinhole()
    epipolar()
    triangulation()
    depth_vs_pointmap()
    self_vs_cross_attention()
    print("done")
