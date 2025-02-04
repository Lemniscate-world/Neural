import os
import datetime

RESEARCH_TEMPLATE = """
    \\documentclass{article}
    \\usepackage{graphicx}
    \\usepackage{amsmath}
    \\usepackage{hyperref}

    \\title{{{title}}}
    \\author{{{author}}}
    \\date{{{date}}}

    \\begin{document}

    \\maketitle

    \\section{Abstract}
    This paper presents an in-depth analysis of the model \\textbf{{{model_name}}}, trained using the \\textbf{Neural Framework}. We provide architecture details, training configurations, benchmarks, and performance evaluations.

    \\section{Introduction}
    Deep learning models have advanced significantly in recent years. This work documents the training process, evaluation metrics, and insights for the \\textbf{{{model_name}}} model.

    \\section{Model Architecture}
    The model consists of the following layers:
    \\begin{itemize}
    {layer_details}
    \\end{itemize}

    \\section{Training Configuration}
    \\begin{itemize}
    \\item Loss Function: \\textbf{{{loss_function}}}
    \\item Optimizer: \\textbf{{{optimizer}}}
    \\item Device: \\textbf{{{device}}}
    \\item Training Time: \\textbf{{{training_time}}}
    \\end{itemize}

    \\section{Benchmark Results}
    Model performance was evaluated on multiple datasets, with the following metrics:
    \\begin{itemize}
    \\item Accuracy: \\textbf{{{accuracy}}}%
    \\item Precision: \\textbf{{{precision}}}%
    \\item Recall: \\textbf{{{recall}}}%
    \\item F1 Score: \\textbf{{{f1_score}}}
    \\end{itemize}

    \\section{Conclusion}
    This study demonstrates the effectiveness of the \\textbf{{{model_name}}} model. Future work will explore hyperparameter tuning and model compression for real-time inference.

    \\bibliographystyle{plain}
    \\bibliography{references}

    \\end{document}
"""

def generate_research_paper(model_data, results):
    """ Generates a research paper in LaTeX format based on the model and training stats. """
    model_name = model_data["name"]
    title = f"Training and Evaluation of {model_name}"
    author = "Neural Research Team"
    date = datetime.date.today().strftime("%B %d, %Y")

    # Format layer details
    layer_details = "\n".join([f"\\item {layer['type']} ({layer.get('params', {})})" for layer in model_data["layers"]])

    latex_content = RESEARCH_TEMPLATE.format(
        title=title,
        author=author,
        date=date,
        model_name=model_name,
        layer_details=layer_details,
        loss_function=model_data["loss"]["value"],
        optimizer=model_data["optimizer"]["value"],
        device=model_data["execution"]["device"],
        training_time=results.get("training_time", "N/A"),
        accuracy=results.get("accuracy", "N/A"),
        precision=results.get("precision", "N/A"),
        recall=results.get("recall", "N/A"),
        f1_score=results.get("f1_score", "N/A"),
    )

    filename = f"{model_name}_paper.tex"
    with open(filename, "w") as file:
        file.write(latex_content)
    
    print(f"Research paper saved as {filename}. Compile with LaTeX.")
