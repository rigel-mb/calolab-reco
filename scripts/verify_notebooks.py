"""Execute local notebooks; compile and validate notebooks requiring Colab CUDA."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / "notebooks").glob("*.ipynb"))
    if not paths:
        raise RuntimeError("No notebooks found")
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell{index}", "exec")
        nbformat.validate(notebook)
        if notebook.metadata.get("execution_environment") == "colab_cuda":
            print(f"Compiled and validated only (requires Colab CUDA): {path.name}")
            continue
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(project)}},
            record_timing=False,
        ).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Executed and verified: {path.name}")


if __name__ == "__main__":
    main()
