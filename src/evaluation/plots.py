"""
Plotting and visualization routines for EviGuard-RAG evaluations.

Implements enterprise cybersecurity XAI figures using Plotly:
1. 2D PCA/SVD embedding manifold with decision boundaries and vector links
2. 5-Axis anomaly radar/spider chart per candidate document
3. Security-Utility Frontier (ASR vs. Clean QA Accuracy across threshold tau)
4. Red-team ablation grouped bar charts and heatmap matrices
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAS_PLOTLY = False

from src.schemas import RetrievalPool, RetrievedCandidate, TrustStatus


# Enterprise Cyber/XAI Theme Palette
COLOR_SAFE = "#10B981"         # Emerald
COLOR_PROMOTED = "#059669"     # Deep Emerald
COLOR_UNCERTAIN = "#F59E0B"    # Amber
COLOR_QUARANTINED = "#EF4444"  # Crimson
COLOR_HIGH_RISK = "#DC2626"    # Deep Crimson
COLOR_RESERVE = "#06B6D4"      # Cyan
COLOR_QUERY = "#8B5CF6"        # Violet
COLOR_SLATE = "#64748B"        # Slate background
COLOR_CARD_BORDER = "rgba(255, 255, 255, 0.08)"


def _get_status_color(status: TrustStatus) -> str:
    """Map candidate TrustStatus to enterprise neon color palette."""
    if status in (TrustStatus.SAFE, TrustStatus.PROMOTED_SAFE):
        return COLOR_SAFE
    if status == TrustStatus.UNCERTAIN:
        return COLOR_UNCERTAIN
    if status in (TrustStatus.QUARANTINED, TrustStatus.HIGH_RISK):
        return COLOR_QUARANTINED
    if status == TrustStatus.REPLACED_RESERVE:
        return COLOR_RESERVE
    return COLOR_SLATE


class DefensePlotter:
    """Generates enterprise-grade Plotly figures for embedding geometric forensics,

    XAI document scorecards, and red-team evaluation frontiers.
    """

    def __init__(self):
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for DefensePlotter: pip install plotly")
        self.default_layout = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#E2E8F0"),
            margin=dict(l=30, r=30, t=50, b=30),
        )

    def plot_embedding_space_2d(self, pool: RetrievalPool) -> go.Figure:
        """Project query and retrieved candidates into 2D via PCA/SVD.

        Features:
        - Query marker in vibrant violet with glow
        - Clean vs. Poison ground truth distinguished by marker symbol
        - Candidate color coded by TrustStatus
        - Translucent decision boundary circle around clean consensus centroid
        - Dashed vector rays from Query to top-ranked candidates
        - Rich hover tooltips with document ID, rank discrepancy, risk, and text snippet
        """
        all_cands = pool.active_set + pool.reserve_pool
        if not all_cands:
            fig = go.Figure()
            fig.update_layout(title="No candidates to visualize", **self.default_layout)
            return fig

        # Prepare 2D projection matrix: [Query, Cand_1, ..., Cand_N]
        embs = [pool.query_embedding] + [c.embedding for c in all_cands]
        X = np.stack(embs, axis=0)

        # 2D PCA / SVD projection
        X_centered = X - np.mean(X, axis=0)
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_centered)
        except Exception:
            # Fallback to pure NumPy SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            coords = U[:, :2] * S[:2]

        q_coord = coords[0]
        cand_coords = coords[1:]

        fig = go.Figure()

        # 1. Dashed vector rays from Query to top 3 active candidates
        for i in range(min(3, len(pool.active_set))):
            c_coord = cand_coords[i]
            fig.add_trace(
                go.Scatter(
                    x=[q_coord[0], c_coord[0]],
                    y=[q_coord[1], c_coord[1]],
                    mode="lines",
                    line=dict(color="rgba(139, 92, 246, 0.35)", width=1.5, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 2. Decision boundary ring around surviving clean consensus centroid
        clean_coords = [
            cand_coords[i]
            for i, c in enumerate(all_cands[:len(pool.active_set)])
            if not (c.doc and c.doc.is_poisoned)
        ]
        if clean_coords:
            center_x = float(np.mean([c[0] for c in clean_coords]))
            center_y = float(np.mean([c[1] for c in clean_coords]))
            max_r = max(
                float(np.sqrt((c[0] - center_x) ** 2 + (c[1] - center_y) ** 2))
                for c in clean_coords
            )
            r = max(max_r * 1.35, 0.25)
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=center_x - r,
                y0=center_y - r,
                x1=center_x + r,
                y1=center_y + r,
                fillcolor="rgba(16, 185, 129, 0.08)",
                line=dict(color="rgba(16, 185, 129, 0.50)", width=1.5, dash="dot"),
            )
            fig.add_annotation(
                x=center_x,
                y=center_y + r * 1.05,
                text="Clean Evidence Consensus Subspace",
                showarrow=False,
                font=dict(size=10, color="rgba(16, 185, 129, 0.8)"),
            )

        # 3. Query marker
        fig.add_trace(
            go.Scatter(
                x=[q_coord[0]],
                y=[q_coord[1]],
                mode="markers+text",
                name="Target Query",
                text=["QUERY"],
                textposition="top center",
                marker=dict(
                    symbol="diamond",
                    size=16,
                    color=COLOR_QUERY,
                    line=dict(color="#FFFFFF", width=2),
                ),
                hovertemplate=(
                    f"<b>[QUERY]</b><br>"
                    f"ID: {pool.query.query_id}<br>"
                    f"Question: {pool.query.question}<br>"
                    f"<extra></extra>"
                ),
            )
        )

        # 4. Group candidates by category for clean legend
        categories = [
            ("Safe Evidence", TrustStatus.SAFE, "circle"),
            ("Promoted Reserve", TrustStatus.REPLACED_RESERVE, "hexagon"),
            ("Uncertain Triage", TrustStatus.UNCERTAIN, "square"),
            ("Quarantined Poison", TrustStatus.QUARANTINED, "x"),
            ("High Risk Outlier", TrustStatus.HIGH_RISK, "cross"),
        ]

        for label, status_filter, sym in categories:
            xs, ys, hover_texts, colors, sizes, symbols = [], [], [], [], [], []
            for i, cand in enumerate(all_cands):
                effective_status = cand.status
                if effective_status != status_filter:
                    # Map promoted safe to safe category
                    if status_filter == TrustStatus.SAFE and effective_status == TrustStatus.PROMOTED_SAFE:
                        pass
                    else:
                        continue

                is_p = bool(cand.doc and cand.doc.is_poisoned)
                coord = cand_coords[i]
                xs.append(coord[0])
                ys.append(coord[1])
                colors.append(_get_status_color(effective_status))
                sizes.append(14 if is_p else 10)
                symbols.append("x" if is_p else sym)

                tag = "⚠️ ADVERSARIAL POISON" if is_p else "✅ CLEAN CONSENSUS"
                hover_text = (
                    f"<b>{cand.doc.doc_id}</b> — {tag}<br>"
                    f"Title: {cand.doc.title[:45]}<br>"
                    f"Status: <b>{effective_status.value.upper()}</b><br>"
                    f"Dense Rank: #{cand.dense_rank} | BM25 Rank: #{cand.bm25_rank}<br>"
                    f"Dense Similarity: {cand.dense_score:.3f}<br>"
                    f"Stage 1 Risk: {cand.stage1_risk:.3f}<br>"
                    f"Final Risk: {cand.final_risk:.3f}<br>"
                    f"Text: {cand.doc.text[:75]}..."
                )
                hover_texts.append(hover_text)

            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=label,
                        marker=dict(
                            symbol=symbols,
                            size=sizes,
                            color=colors,
                            line=dict(color="#FFFFFF", width=1),
                        ),
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_texts,
                    )
                )

        fig.update_layout(
            title=dict(
                text="🌌 2D Embedding Manifold & Vector Geometry (PCA)",
                font=dict(size=15, color="#F8FAFC"),
            ),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(size=11),
            ),
            **self.default_layout,
        )
        return fig

    def plot_candidate_radar(self, cand: RetrievedCandidate) -> go.Figure:
        """Generate a 5-axis normalized anomaly spider/radar chart for a single candidate.

        Axes:
        1. LOO Geometry (G_i)
        2. Local Isolation (L_i)
        3. Hybrid Disagreement (D_i)
        4. Sentence Drift (S_drift)
        5. NLI Contradiction (S_contra)
        """
        # Normalization to [0, 1] range for visual comparability
        def norm_z(z: float) -> float:
            return float(1.0 / (1.0 + np.exp(-np.clip(z, -6.0, 6.0))))

        g_norm = norm_z(cand.loo_geometric_score)
        l_norm = norm_z(cand.local_connectivity_score)
        d_norm = float(np.clip(cand.hybrid_disagreement_score, 0.0, 1.0))
        drift_norm = float(np.clip(cand.sentence_drift_score or 0.0, 0.0, 1.0))
        nli_norm = float(np.clip(cand.nli_contradiction_score or 0.0, 0.0, 1.0))

        categories = [
            "LOO Geometry<br>(G_i)",
            "Local Isolation<br>(L_i)",
            "Hybrid Disagreement<br>(D_i)",
            "Sentence Drift<br>(S_drift)",
            "NLI Contradiction<br>(S_contra)",
        ]
        values = [g_norm, l_norm, d_norm, drift_norm, nli_norm]
        # Close the loop
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        status_color = _get_status_color(cand.status)
        fill_color = (
            "rgba(239, 68, 68, 0.25)" if cand.status in (TrustStatus.QUARANTINED, TrustStatus.HIGH_RISK)
            else "rgba(16, 185, 129, 0.25)"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=status_color, width=2.5),
                marker=dict(size=6, color=status_color),
                name=cand.doc.doc_id,
            )
        )

        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.0],
                    gridcolor="rgba(255, 255, 255, 0.10)",
                    tickvals=[0.25, 0.50, 0.75, 1.0],
                    tickfont=dict(size=8, color="#94A3B8"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255, 255, 255, 0.10)",
                    tickfont=dict(size=10, color="#CBD5E1"),
                ),
            ),
            showlegend=False,
            height=280,
            **self.default_layout,
        )
        return fig

    def plot_security_utility_frontier(self, sweep_data: Dict[str, Any]) -> go.Figure:
        """Interactive line chart demonstrating ASR reduction vs Clean QA Accuracy across threshold tau."""
        thresholds = sweep_data.get("thresholds", [])
        asr = [v * 100.0 for v in sweep_data.get("asr_curve", [])]
        acc = [v * 100.0 for v in sweep_data.get("acc_curve", [])]

        fig = go.Figure()

        # Shaded recommended operating zone
        tau_low = sweep_data.get("tau_low_recommended", 0.35)
        tau_high = sweep_data.get("tau_high_recommended", 0.75)
        fig.add_vrect(
            x0=tau_low,
            x1=tau_high,
            fillcolor="rgba(16, 185, 129, 0.12)",
            line=dict(color="rgba(16, 185, 129, 0.40)", width=1, dash="dot"),
            annotation_text="Optimal Operating Zone (tau: 0.35 - 0.75)",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#10B981"),
        )

        # Attack Success Rate (Crimson)
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=asr,
                name="Attack Success Rate (ASR %)",
                mode="lines+markers",
                line=dict(color=COLOR_QUARANTINED, width=3),
                marker=dict(size=7, symbol="circle"),
                hovertemplate="Threshold tau: %{x}<br>ASR: %{y:.1f}%<extra></extra>",
            )
        )

        # Clean QA Accuracy (Emerald)
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=acc,
                name="Clean QA Accuracy (%)",
                mode="lines+markers",
                line=dict(color=COLOR_SAFE, width=3),
                marker=dict(size=7, symbol="diamond"),
                hovertemplate="Threshold tau: %{x}<br>Clean Accuracy: %{y:.1f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="📊 Security-Utility Operating Frontier",
                font=dict(size=15, color="#F8FAFC"),
            ),
            xaxis=dict(
                title="Defense Threshold tau",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
            ),
            yaxis=dict(
                title="Percentage (%)",
                range=[0, 105],
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
            ),
            **self.default_layout,
        )
        return fig

    def plot_ablation_bars(self, ablation_data: Dict[str, Any]) -> go.Figure:
        """Grouped bar chart comparing ASR % across defense versions for each attack tier."""
        tier_labels = ablation_data.get("tier_labels", [])
        versions = ablation_data.get("versions", [])
        asr_matrix = np.array(ablation_data.get("asr_matrix", []))

        palette = [COLOR_QUARANTINED, COLOR_UNCERTAIN, COLOR_RESERVE, COLOR_SAFE]

        fig = go.Figure()
        for col_idx, (version_name, color) in enumerate(zip(versions, palette)):
            values = asr_matrix[:, col_idx] if len(asr_matrix) > 0 else []
            fig.add_trace(
                go.Bar(
                    x=tier_labels,
                    y=values,
                    name=version_name,
                    marker=dict(color=color, opacity=0.85),
                    hovertemplate=f"<b>{version_name}</b><br>Tier: %{{x}}<br>ASR: %{{y:.1f}}%<extra></extra>",
                )
            )

        fig.update_layout(
            barmode="group",
            title=dict(
                text="🛡️ Attack Success Rate (ASR %) across Defense Generations",
                font=dict(size=15, color="#F8FAFC"),
            ),
            yaxis=dict(
                title="Attack Success Rate (ASR %)",
                range=[0, 100],
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
            ),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            **self.default_layout,
        )
        return fig

    def plot_ablation_heatmap(self, ablation_data: Dict[str, Any]) -> go.Figure:
        """Matrix heatmap showing Attack-Tier x Defense-Version ASR mitigation."""
        tier_labels = ablation_data.get("tier_labels", [])
        versions = ablation_data.get("versions", [])
        asr_matrix = ablation_data.get("asr_matrix", [])

        # Color scale: Emerald (low ASR) to Crimson (high ASR)
        colorscale = [
            [0.0, "#10B981"],
            [0.2, "#059669"],
            [0.4, "#F59E0B"],
            [0.7, "#EF4444"],
            [1.0, "#991B1B"],
        ]

        text_annotations = [
            [f"{val:.1f}%" for val in row]
            for row in asr_matrix
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=asr_matrix,
                x=versions,
                y=tier_labels,
                text=text_annotations,
                texttemplate="%{text}",
                textfont=dict(size=13, color="#FFFFFF", family="JetBrains Mono, monospace"),
                colorscale=colorscale,
                reversescale=False,
                colorbar=dict(title="ASR %", ticksuffix="%"),
                hovertemplate="<b>%{y}</b><br>%{x}<br>ASR: %{z:.1f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="🔥 Defense Ablation Matrix: ASR (%) by Attack Tier",
                font=dict(size=15, color="#F8FAFC"),
            ),
            **self.default_layout,
        )
        return fig

    def plot_ablation_heatmap_and_bars(self, ablation_data: Dict[str, Any]) -> go.Figure:
        """Returns the primary ablation bar chart (or composite view)."""
        return self.plot_ablation_bars(ablation_data)
