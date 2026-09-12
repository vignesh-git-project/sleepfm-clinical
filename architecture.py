import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_architecture(output_path="sleepfm_neural_architecture.png"):
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")

    # Helper function to create clean styled network blocks
    def draw_box(x, y, w, h, title, subtitle="", color="#1e293b", edge="#38bdf8", text_color="#f8fafc"):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color, edgecolor=edge, linewidth=1.5
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                    color=text_color, fontsize=10.5, fontweight="bold")
            ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                    color="#94a3b8", fontsize=8.5, family="monospace")
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    color=text_color, fontsize=10.5, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, label="", color="#38bdf8", rad=0.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.8,
                shrinkA=3, shrinkB=3,
                connectionstyle=f"arc3,rad={rad}"
            )
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                    color="#cbd5e1", fontsize=8, family="monospace")

    # Titles & Section Headers
    ax.text(0.5, 9.5, "SleepFM Foundation Model: Multi-Modal Edge Architecture",
            ha="left", va="center", color="#38bdf8", fontsize=18, fontweight="bold")
    ax.text(0.5, 9.1, "Hierarchical Sequence Representation & Downstream Multi-Task Disentanglement",
            ha="left", va="center", color="#94a3b8", fontsize=11)

    # Column 1: Input Modalities (C=4)
    draw_box(0.5, 7.3, 2.8, 0.9, "MPU6050 (Actigraphy)", "BAS: (3,686,400,) @ 128Hz", "#111827", "#10b981")
    draw_box(0.5, 5.7, 2.8, 0.9, "MAX30100 (Optical PPG)", "RESP: (3,686,400,) @ 128Hz", "#111827", "#10b981")
    draw_box(0.5, 4.1, 2.8, 0.9, "AD8232 (Single ECG)", "EKG: (3,686,400,) @ 128Hz", "#111827", "#10b981")
    draw_box(0.5, 2.5, 2.8, 0.9, "Submental EMG (Absent)", "EMG: Zero-Padded Tensor", "#1e1e24", "#ef4444", text_color="#ef4444")

    # Column 2: SetTransformer Patch Projection (Stage 1)
    draw_box(4.2, 7.3, 2.6, 0.9, "1D Patch Conv + MHSA", "Conv1D(640,128) | 8-Heads", "#1e293b", "#38bdf8")
    draw_box(4.2, 5.7, 2.6, 0.9, "1D Patch Conv + MHSA", "Conv1D(640,128) | 8-Heads", "#1e293b", "#38bdf8")
    draw_box(4.2, 4.1, 2.6, 0.9, "1D Patch Conv + MHSA", "Conv1D(640,128) | 8-Heads", "#1e293b", "#38bdf8")
    draw_box(4.2, 2.5, 2.6, 0.9, "Attention Key Masking", "M[:, 3, :] = True (-inf)", "#1e1e24", "#ef4444", text_color="#ef4444")

    for y in [7.75, 6.15, 4.55]:
        draw_arrow(3.3, y, 4.2, y, "640 smp")
    draw_arrow(3.3, 2.95, 4.2, 2.95, "Mask bit", color="#ef4444")

    # Column 3: Cross-Modal Stacking Tensor
    draw_box(7.7, 4.1, 2.2, 4.1, "Cross-Modal Stacking\n& Mask Injection",
             "X: (1, 4, 5760, 128)\nM: (1, 4, 5760)", "#0f172a", "#818cf8")

    draw_arrow(6.8, 7.75, 7.7, 6.8, "(5760, 128)", rad=-0.1)
    draw_arrow(6.8, 6.15, 7.7, 6.2, "(5760, 128)")
    draw_arrow(6.8, 4.55, 7.7, 5.6, "(5760, 128)", rad=0.1)
    draw_arrow(6.8, 2.95, 7.7, 4.8, "Masked", color="#ef4444", rad=0.15)

    # Bifurcation to Downstream Stage 2 Heads
    # Top Branch: Stage 2A Sleep Staging
    draw_box(10.8, 6.8, 2.5, 0.9, "Spatial Attention Pool", "Re-weights across C=4", "#1e293b", "#38bdf8")
    draw_box(10.8, 5.3, 2.5, 0.9, "Temporal Bi-LSTM", "Hidden Dim: 256", "#1e293b", "#38bdf8")
    draw_box(10.8, 3.8, 2.5, 0.9, "30s Epoch Mean-Pool", "Pool 6 tokens (6x5s)", "#1e293b", "#38bdf8")
    draw_box(13.9, 5.3, 2.6, 2.4, "5-Stage AASM Hypnogram",
             "Wake (15.1%)\nN1 (9.8%)\nN2 (16.2%)\nN3 Deep (26.2%)\nREM (32.6%)\nEff: 84.9%",
             "#064e3b", "#34d399")

    draw_arrow(9.9, 6.5, 10.8, 7.25, rad=0.1)
    draw_arrow(12.05, 6.8, 12.05, 6.2)
    draw_arrow(12.05, 5.3, 12.05, 4.7)
    draw_arrow(13.3, 4.25, 13.9, 6.1, "960 Epochs", rad=0.1)

    # Bottom Branch: Stage 2B Cox Survival Heads
    draw_box(10.8, 2.2, 2.5, 0.9, "Global Temporal Attn", "Whole-Night Vector: (1, 128)", "#1e293b", "#f59e0b")
    draw_box(10.8, 0.7, 2.5, 0.9, "Demographic Prior Fusion", "Age: -2.67 | Sex: 1.0 (1, 130)", "#1e293b", "#f59e0b")
    draw_box(13.9, 0.7, 2.6, 2.4, "1,065 Linear Cox Heads",
             "OSA: 70.8th %tile\nCSA: 67.3rd %tile\nCHF: 77.5th %tile\nHTN: 22.3rd %tile\nT2D: 24.9th %tile",
             "#451a03", "#fbbf24")

    draw_arrow(9.9, 5.5, 10.8, 2.65, rad=-0.2)
    draw_arrow(12.05, 2.2, 12.05, 1.6)
    draw_arrow(13.3, 1.15, 13.9, 1.5, "130-dim vector")

    # Bounds and formatting
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[*] Architecture diagram generated: {output_path}")

if __name__ == "__main__":
    draw_neural_architecture()
