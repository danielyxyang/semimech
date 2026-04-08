# Semi-Mechanistic Residual Stream Analysis

A comprehensive suite for extracting, analyzing, and visualizing internal **activation trajectories** from the residual streams of large language models. It identifies "concept directions" using readers (such as PCA or Linear Probing), projecting activations to visualize how internal representations evolve across layers and token positions.

## Features

- **Activation Extraction:** Easily extract hidden states and top-k next-token predictions from Hugging Face models using standard or custom datasets.
- **Interactive GUI:** Jupyter widget-based interface for instant exploration without writing code.
- **Manual API:** Flexible code-based workflow for batch processing, customized analysis, and fine-grained control over pooling and reading methods.
- **Rich Visualizations:** Interactive Plotly graphs for per-layer and per-token trajectory analysis, complete with rich context tooltips and vocabulary embeddings.

## Setup

We recommend using [uv](https://github.com/astral-sh/uv) to install the package into your environment:
```bash
uv pip install -e .
```

## Usage

The primary entry point is the notebook [`residual_stream_analysis.ipynb`](notebooks/residual_stream_analysis.ipynb)

### Interactive Analysis
This notebook provides an interactive GUI that lets you:
1. **Extract activations** by selecting a model, dataset, and custom responses.
2. **Analyze & Visualize** by clicking through per-layer or per-token analyses, choosing reading methods, pooling methods, and utilizing interactive plots with zooming and rich tooltips.

### Manual Analysis API
The second half of this notebook serves as a getting-started tutorial for the `semimech` Python API. It demonstrates how to programmatically:
- Load models and datasets (`load_model_and_tokenizer_from_spec`, `load_dataset_from_spec`).
- Extract, save, and load activations (`extract_activations`, `Activations.save`, `Activations.load`).
- Analyze per-layer and per-token trajectories (`analyze_per_layer`, `analyze_per_token`).
- Visualize trajectories (`plot_per_layer`, `plot_per_token`).

## License

All content in this repository is licensed under the [MIT license](LICENSE).