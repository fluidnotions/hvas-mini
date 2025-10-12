# Agent Task: Enhanced Visualization (M5)

## Branch: `feature/visualization-v2`

## Priority: MEDIUM (sequential after M1-M4)

## Execution: SEQUENTIAL (depends on M1-M4)

## Objective

Create comprehensive visualizations showing emergent hierarchical organization through embedding space plots, weight matrix heatmaps, topology evolution graphs, and memory age distributions.

**Current**: Rich terminal UI with basic status tables
**Target**: Multi-dimensional visualization revealing system dynamics and self-organization

## Dependencies

- ✅ M1 (async-orchestration) - provides timing data
- ✅ M2 (agent-weighting) - provides weight matrices
- ✅ M3 (memory-decay) - provides memory age data
- ✅ M4 (meta-agent) - provides topology evolution
- ALL previous milestones MUST be merged before starting M5

## Background

Current visualization (iteration 1) shows:
- Agent status and scores
- Memory retrieval counts
- Parameter evolution
- Stream logs

Missing critical insights:
- Agent positioning in embedding space (clustering patterns)
- Trust network structure (weight relationships)
- Graph topology changes over time
- Memory freshness distribution
- Emergent organization patterns

## Tasks

### 1. Add Visualization Dependencies

**File**: `pyproject.toml`

Add visualization libraries:

```toml
[project]
dependencies = [
    # ... existing dependencies ...

    # NEW: Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "networkx>=3.0",
    "scikit-learn>=1.3.0",  # For PCA/t-SNE
]

[project.optional-dependencies]
viz = [
    "umap-learn>=0.5.0",  # Better than PCA, but optional
]
```

### 2. Create Visualization Package Structure

**New Directory**: `src/hvas_mini/visualization/`

Refactor existing visualization.py into a package:

**File**: `src/hvas_mini/visualization/__init__.py`
```python
"""
Enhanced visualization suite for HVAS Mini.
"""

from hvas_mini.visualization.stream_visualizer import StreamVisualizer
from hvas_mini.visualization.embedding_plot import EmbeddingPlotter
from hvas_mini.visualization.weight_heatmap import WeightHeatmap
from hvas_mini.visualization.topology_view import TopologyViewer
from hvas_mini.visualization.memory_age_plot import MemoryAgePlotter

__all__ = [
    "StreamVisualizer",
    "EmbeddingPlotter",
    "WeightHeatmap",
    "TopologyViewer",
    "MemoryAgePlotter",
]
```

**File**: `src/hvas_mini/visualization/stream_visualizer.py`
```python
"""
Terminal-based streaming visualizer (existing functionality).
"""

# Move existing StreamVisualizer class here from old visualization.py
# No changes needed - just reorganization
```

### 3. Create Embedding Space Visualization

**File**: `src/hvas_mini/visualization/embedding_plot.py`
```python
"""
Agent embedding space visualization using dimensionality reduction.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List
import os

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class EmbeddingPlotter:
    """Visualizes agent outputs in 2D embedding space."""

    def __init__(self, use_umap: bool = False):
        """Initialize embedding plotter.

        Args:
            use_umap: Use UMAP instead of PCA (requires umap-learn)
        """
        self.use_umap = use_umap and HAS_UMAP
        self.reducer = None
        self.embedding_history: List[Dict] = []

    def plot_agent_embeddings(
        self,
        agent_outputs: Dict[str, str],
        embeddings: Dict[str, np.ndarray],
        generation: int,
        save_path: str | None = None,
    ):
        """Plot agent outputs in 2D embedding space.

        Args:
            agent_outputs: {agent_name: output_text}
            embeddings: {agent_name: embedding_vector}
            generation: Current generation number
            save_path: Optional path to save plot
        """
        if len(embeddings) < 2:
            return

        # Prepare data
        agents = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[agent] for agent in agents])

        # Reduce to 2D
        if self.use_umap:
            if self.reducer is None:
                self.reducer = UMAP(n_components=2, random_state=42)
            coords_2d = self.reducer.fit_transform(embedding_matrix)
        else:
            if self.reducer is None:
                self.reducer = PCA(n_components=2, random_state=42)
            coords_2d = self.reducer.fit_transform(embedding_matrix)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        colors = plt.cm.tab10(range(len(agents)))
        for i, agent in enumerate(agents):
            ax.scatter(
                coords_2d[i, 0],
                coords_2d[i, 1],
                c=[colors[i]],
                s=200,
                alpha=0.7,
                label=agent,
            )

            # Annotate
            ax.annotate(
                agent,
                (coords_2d[i, 0], coords_2d[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
            )

        ax.set_title(
            f"Agent Embedding Space (Generation {generation})\n"
            f"Method: {'UMAP' if self.use_umap else 'PCA'}",
            fontsize=14,
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

        # Store in history
        self.embedding_history.append({
            "generation": generation,
            "agents": agents,
            "coords": coords_2d.tolist(),
        })

    def plot_embedding_evolution(
        self,
        save_path: str | None = None,
    ):
        """Plot how agent positions change over generations.

        Args:
            save_path: Optional path to save plot
        """
        if len(self.embedding_history) < 2:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Get all unique agents
        all_agents = set()
        for record in self.embedding_history:
            all_agents.update(record["agents"])

        colors = {agent: plt.cm.tab10(i) for i, agent in enumerate(all_agents)}

        # Plot trajectories
        for agent in all_agents:
            trajectory_x = []
            trajectory_y = []

            for record in self.embedding_history:
                if agent in record["agents"]:
                    idx = record["agents"].index(agent)
                    trajectory_x.append(record["coords"][idx][0])
                    trajectory_y.append(record["coords"][idx][1])

            if trajectory_x:
                ax.plot(
                    trajectory_x,
                    trajectory_y,
                    c=colors[agent],
                    alpha=0.6,
                    linewidth=2,
                    marker="o",
                    markersize=6,
                    label=agent,
                )

                # Annotate start and end
                ax.text(
                    trajectory_x[0],
                    trajectory_y[0],
                    "Start",
                    fontsize=8,
                    alpha=0.7,
                )
                ax.text(
                    trajectory_x[-1],
                    trajectory_y[-1],
                    "End",
                    fontsize=8,
                    alpha=0.7,
                )

        ax.set_title("Agent Embedding Trajectories Across Generations", fontsize=14)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
```

### 4. Create Weight Matrix Heatmap

**File**: `src/hvas_mini/visualization/weight_heatmap.py`
```python
"""
Trust weight matrix heatmap visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict


class WeightHeatmap:
    """Visualizes agent trust weight matrices as heatmaps."""

    def __init__(self):
        """Initialize weight heatmap plotter."""
        self.weight_history: list = []

    def plot_weight_matrix(
        self,
        agent_weights: Dict[str, Dict[str, float]],
        generation: int,
        save_path: str | None = None,
    ):
        """Plot trust weight matrix as heatmap.

        Args:
            agent_weights: {agent: {peer: trust_weight}}
            generation: Current generation number
            save_path: Optional path to save plot
        """
        if not agent_weights:
            return

        # Build matrix
        agents = sorted(agent_weights.keys())
        n = len(agents)
        matrix = np.zeros((n, n))

        for i, agent in enumerate(agents):
            for j, peer in enumerate(agents):
                if agent == peer:
                    matrix[i, j] = np.nan  # Diagonal = self-trust (undefined)
                else:
                    matrix[i, j] = agent_weights.get(agent, {}).get(peer, 0.5)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            xticklabels=agents,
            yticklabels=agents,
            cbar_kws={"label": "Trust Weight"},
            ax=ax,
        )

        ax.set_title(
            f"Agent Trust Weight Matrix (Generation {generation})",
            fontsize=14,
        )
        ax.set_xlabel("To Agent (Peer)")
        ax.set_ylabel("From Agent (Observer)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

        # Store in history
        self.weight_history.append({
            "generation": generation,
            "agents": agents,
            "matrix": matrix.copy(),
        })

    def plot_weight_evolution(
        self,
        agent_pair: tuple[str, str],
        save_path: str | None = None,
    ):
        """Plot how trust weight evolves for a specific agent pair.

        Args:
            agent_pair: (from_agent, to_agent)
            save_path: Optional path to save plot
        """
        if len(self.weight_history) < 2:
            return

        from_agent, to_agent = agent_pair

        generations = []
        weights = []

        for record in self.weight_history:
            if from_agent in record["agents"] and to_agent in record["agents"]:
                from_idx = record["agents"].index(from_agent)
                to_idx = record["agents"].index(to_agent)

                generations.append(record["generation"])
                weights.append(record["matrix"][from_idx, to_idx])

        if not generations:
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(generations, weights, marker="o", linewidth=2, markersize=8)

        ax.set_title(
            f"Trust Weight Evolution: {from_agent} → {to_agent}",
            fontsize=14,
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Trust Weight")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
```

### 5. Create Topology Graph Visualization

**File**: `src/hvas_mini/visualization/topology_view.py`
```python
"""
Graph topology visualization using networkx.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List


class TopologyViewer:
    """Visualizes agent graph topology and its evolution."""

    def __init__(self):
        """Initialize topology viewer."""
        self.topology_snapshots: List[Dict] = []

    def plot_topology(
        self,
        active_agents: List[str],
        agent_weights: Dict[str, Dict[str, float]] | None = None,
        generation: int = 0,
        save_path: str | None = None,
    ):
        """Plot current graph topology.

        Args:
            active_agents: List of active agent names
            agent_weights: Optional trust weights for edge weights
            generation: Current generation number
            save_path: Optional path to save plot
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for agent in active_agents:
            G.add_node(agent)

        # Add edges (trust relationships)
        if agent_weights:
            for from_agent, peers in agent_weights.items():
                for to_agent, weight in peers.items():
                    if from_agent in active_agents and to_agent in active_agents:
                        G.add_edge(from_agent, to_agent, weight=weight)

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw nodes
        node_colors = plt.cm.tab10(range(len(active_agents)))
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.8,
            ax=ax,
        )

        # Draw edges with weights
        if agent_weights:
            edges = G.edges(data=True)
            edge_weights = [edge[2]["weight"] for edge in edges]

            nx.draw_networkx_edges(
                G,
                pos,
                width=[w * 3 for w in edge_weights],  # Scale width by weight
                alpha=0.5,
                edge_color=edge_weights,
                edge_cmap=plt.cm.RdYlGn,
                edge_vmin=0.0,
                edge_vmax=1.0,
                arrows=True,
                arrowsize=20,
                ax=ax,
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos,
                arrows=True,
                arrowsize=20,
                ax=ax,
            )

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=12,
            font_weight="bold",
            ax=ax,
        )

        ax.set_title(
            f"Agent Graph Topology (Generation {generation})\n"
            f"{len(active_agents)} agents, {G.number_of_edges()} relationships",
            fontsize=14,
        )
        ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

        # Store snapshot
        self.topology_snapshots.append({
            "generation": generation,
            "agents": active_agents.copy(),
            "graph": G.copy(),
        })

    def plot_topology_evolution(self, save_path: str | None = None):
        """Plot topology changes across generations.

        Creates a grid of topology snapshots showing evolution.

        Args:
            save_path: Optional path to save plot
        """
        if len(self.topology_snapshots) < 2:
            return

        # Create grid
        n_snapshots = min(6, len(self.topology_snapshots))  # Max 6 snapshots
        indices = np.linspace(0, len(self.topology_snapshots) - 1, n_snapshots, dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            snapshot = self.topology_snapshots[idx]
            G = snapshot["graph"]
            generation = snapshot["generation"]

            pos = nx.spring_layout(G, seed=42)

            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color=range(G.number_of_nodes()),
                node_size=1500,
                cmap=plt.cm.tab10,
                font_size=10,
                font_weight="bold",
                arrows=True,
                ax=axes[i],
            )

            axes[i].set_title(f"Generation {generation}", fontsize=12)

        # Hide unused subplots
        for i in range(n_snapshots, len(axes)):
            axes[i].axis("off")

        fig.suptitle("Graph Topology Evolution", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
```

### 6. Create Memory Age Distribution Plot

**File**: `src/hvas_mini/visualization/memory_age_plot.py`
```python
"""
Memory age distribution visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Dict


class MemoryAgePlotter:
    """Visualizes memory age distributions."""

    def plot_age_distribution(
        self,
        memory_stats: Dict[str, Dict],
        save_path: str | None = None,
    ):
        """Plot memory age distribution for all agents.

        Args:
            memory_stats: {agent_name: stats_dict}
            save_path: Optional path to save plot
        """
        if not memory_stats:
            return

        fig, axes = plt.subplots(
            len(memory_stats),
            1,
            figsize=(10, 4 * len(memory_stats)),
        )

        if len(memory_stats) == 1:
            axes = [axes]

        for i, (agent, stats) in enumerate(memory_stats.items()):
            ages = stats.get("age_distribution", [])

            if not ages:
                axes[i].text(
                    0.5,
                    0.5,
                    "No memory data",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{agent} - Memory Age Distribution")
                continue

            # Create histogram
            axes[i].hist(
                ages,
                bins=20,
                color=plt.cm.tab10(i),
                alpha=0.7,
                edgecolor="black",
            )

            axes[i].set_title(f"{agent} - Memory Age Distribution")
            axes[i].set_xlabel("Age (days)")
            axes[i].set_ylabel("Count")
            axes[i].grid(alpha=0.3, axis="y")

            # Add statistics
            if ages:
                avg_age = np.mean(ages)
                max_age = np.max(ages)
                axes[i].axvline(
                    avg_age,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Avg: {avg_age:.1f} days",
                )
                axes[i].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_memory_growth(
        self,
        memory_counts: List[Dict[str, int]],
        save_path: str | None = None,
    ):
        """Plot memory collection growth over time.

        Args:
            memory_counts: [{generation: N, agent: count}]
            save_path: Optional path to save plot
        """
        if not memory_counts:
            return

        # Organize data by agent
        agents = set()
        for record in memory_counts:
            agents.update(k for k in record.keys() if k != "generation")

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, agent in enumerate(sorted(agents)):
            generations = [r["generation"] for r in memory_counts if agent in r]
            counts = [r[agent] for r in memory_counts if agent in r]

            ax.plot(
                generations,
                counts,
                marker="o",
                linewidth=2,
                markersize=6,
                label=agent,
                color=plt.cm.tab10(i),
            )

        ax.set_title("Memory Collection Growth Over Generations", fontsize=14)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Memory Count")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
```

### 7. Integrate Visualizations into Pipeline

**File**: `src/hvas_mini/pipeline.py`

Add visualization exports after each generation:

```python
import os
from hvas_mini.visualization import (
    EmbeddingPlotter,
    WeightHeatmap,
    TopologyViewer,
    MemoryAgePlotter,
)


class HVASMiniPipeline:
    def __init__(self, persist_directory: str = "./data/memories"):
        # ... existing init ...

        # NEW: Advanced visualizers
        self.embedding_plotter = EmbeddingPlotter(use_umap=False)
        self.weight_heatmap = WeightHeatmap()
        self.topology_viewer = TopologyViewer()
        self.memory_age_plotter = MemoryAgePlotter()

        self.visualization_dir = "./data/visualizations"
        os.makedirs(self.visualization_dir, exist_ok=True)

    async def generate(
        self, topic: str, config: Dict | None = None
    ) -> BlogState:
        """Generate with enhanced visualization exports."""

        # ... existing generation logic ...

        final_state = await self.visualizer.display_stream(state_stream())

        # NEW: Export visualizations after generation
        if os.getenv("EXPORT_VISUALIZATIONS", "false").lower() == "true":
            self._export_visualizations(final_state)

        return final_state

    def _export_visualizations(self, state: BlogState):
        """Export all visualization plots.

        Args:
            state: Final BlogState after generation
        """
        generation = state.get("generation_id", "unknown")

        # 1. Weight matrix heatmap
        if state.get("agent_weights"):
            self.weight_heatmap.plot_weight_matrix(
                state["agent_weights"],
                generation=int(generation.split("_")[-1]) if "_" in generation else 0,
                save_path=f"{self.visualization_dir}/weights_{generation}.png",
            )

        # 2. Topology graph
        if state.get("active_agents"):
            self.topology_viewer.plot_topology(
                state["active_agents"],
                agent_weights=state.get("agent_weights"),
                generation=int(generation.split("_")[-1]) if "_" in generation else 0,
                save_path=f"{self.visualization_dir}/topology_{generation}.png",
            )

        # TODO: 3. Embedding plot (requires agent output embeddings)
        # TODO: 4. Memory age distribution (requires memory stats)
```

### 8. Add Configuration

**File**: `.env.example`

```bash
# Enhanced Visualization Settings
EXPORT_VISUALIZATIONS=false        # Set to true to export plots
VISUALIZATION_DIR=./data/visualizations
USE_UMAP_EMBEDDINGS=false          # Requires umap-learn
```

### 9. Create Tests

**File**: `test_visualization_v2.py`

```python
"""Tests for enhanced visualization."""

import pytest
import numpy as np
from hvas_mini.visualization import (
    EmbeddingPlotter,
    WeightHeatmap,
    TopologyViewer,
    MemoryAgePlotter,
)


def test_embedding_plotter_initialization():
    """Test EmbeddingPlotter initializes correctly."""
    plotter = EmbeddingPlotter(use_umap=False)
    assert plotter.use_umap is False
    assert plotter.embedding_history == []


def test_weight_heatmap_history():
    """Test WeightHeatmap stores history."""
    heatmap = WeightHeatmap()

    weights = {
        "agent1": {"agent2": 0.8},
        "agent2": {"agent1": 0.6},
    }

    heatmap.plot_weight_matrix(weights, generation=1, save_path=None)

    assert len(heatmap.weight_history) == 1
    assert heatmap.weight_history[0]["generation"] == 1


def test_topology_viewer_snapshots():
    """Test TopologyViewer stores snapshots."""
    viewer = TopologyViewer()

    active_agents = ["intro", "body", "conclusion"]
    weights = {
        "body": {"intro": 0.8},
        "conclusion": {"intro": 0.7, "body": 0.9},
    }

    viewer.plot_topology(
        active_agents,
        agent_weights=weights,
        generation=1,
        save_path=None,
    )

    assert len(viewer.topology_snapshots) == 1
    assert viewer.topology_snapshots[0]["generation"] == 1
    assert viewer.topology_snapshots[0]["agents"] == active_agents


def test_memory_age_plotter():
    """Test MemoryAgePlotter handles empty data."""
    plotter = MemoryAgePlotter()

    stats = {}

    # Should not crash on empty data
    plotter.plot_age_distribution(stats, save_path=None)
```

## Deliverables Checklist

- [ ] `pyproject.toml` updated with visualization dependencies
- [ ] `src/hvas_mini/visualization/` package created
- [ ] `src/hvas_mini/visualization/stream_visualizer.py` (refactored from old code)
- [ ] `src/hvas_mini/visualization/embedding_plot.py` created
- [ ] `src/hvas_mini/visualization/weight_heatmap.py` created
- [ ] `src/hvas_mini/visualization/topology_view.py` created
- [ ] `src/hvas_mini/visualization/memory_age_plot.py` created
- [ ] `src/hvas_mini/pipeline.py` integrated with visualization exports
- [ ] `.env.example` updated with visualization settings
- [ ] `test_visualization_v2.py` created with passing tests
- [ ] Sample visualizations generated successfully

## Acceptance Criteria

1. ✅ Embedding space plot shows agent positions in 2D
2. ✅ Weight matrix heatmap displays trust relationships
3. ✅ Topology graph shows active agents and connections
4. ✅ Memory age histogram shows distribution
5. ✅ Evolution plots show changes across generations
6. ✅ All plots exportable to PNG files
7. ✅ All existing tests still pass
8. ✅ New visualization tests pass

## Testing

```bash
cd worktrees/visualization-v2

# Install viz dependencies
uv sync

# Run new visualization tests
uv run pytest test_visualization_v2.py -v

# Run all tests
uv run pytest

# Run demo with visualization export
export ANTHROPIC_API_KEY=your_key
export EXPORT_VISUALIZATIONS=true
uv run python main.py

# Check generated plots
ls data/visualizations/
```

Expected output: PNG files in `data/visualizations/` showing weight matrices, topology graphs, embedding spaces, and memory age distributions.

## Integration Notes

This milestone:
- Integrates data from M1 (timing), M2 (weights), M3 (memory age), M4 (topology)
- Provides comprehensive view of emergent organization
- Enables research analysis of system dynamics
- Creates publication-ready visualizations

## Next Steps

After merging M5 to main:
- Create demo video showing system evolution
- Generate visualization report for documentation
- Publish findings on emergent hierarchical organization
- Iteration 2 COMPLETE! 🎉
