import os
import datetime

RESEARCH_TEMPLATE = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\begin{document}

\maketitle

\section{Abstract}
This paper presents an in-depth analysis of the model \textbf{{{model_name}}}, trained using the \textbf{Neural Framework}. We provide architecture details, training configurations, benchmarks, performance evaluations, and a detailed analysis of shape propagation through the network.

\section{Model Architecture}
The model consists of the following layers:
\begin{itemize}
{layer_details}
\end{itemize}

\section{Shape Propagation}
For an input shape \( (B, H, W, C) \), the transformations are as follows:

\subsection{Convolutional Layers}
For a Conv2D layer with kernel size \( k_h \times k_w \), stride \( s_h \) and \( s_w \), dilation \( d_h \) and \( d_w \), and padding \( p_h \) and \( p_w \):
\[
H_{out} = \frac{H + 2p_h - d_h (k_h - 1) - 1}{s_h} + 1, \quad
W_{out} = \frac{W + 2p_w - d_w (k_w - 1) - 1}{s_w} + 1
\]
\[
C_{out} = \text{filters}
\]

\subsection{Pooling Layers}
\[
H_{out} = \frac{H + 2p_h - k_h}{s_h} + 1, \quad
W_{out} = \frac{W + 2p_w - k_w}{s_w} + 1
\]
\[
C_{out} = C_{in}
\]

\subsection{Dense Layers}
A dense (fully-connected) layer transforms the input as:
\[
(B, D) \rightarrow (B, N)
\]

\subsection{Custom Propagation (if any)}
% Insert custom propagation formulas here

\bigskip
\textbf{Shape Propagation History:}\\
\begin{itemize}
{shape_history_items}
\end{itemize}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{{{shape_prop_img}}}
\caption{Visualization of shape propagation through the network.}
\end{figure}

\section{Training Configuration}
\begin{itemize}
\item Loss Function: \textbf{{{loss_function}}}
\item Optimizer: \textbf{{{optimizer}}}
\item Device: \textbf{{{device}}}
\item Training Time: \textbf{{{training_time}}}
\end{itemize}

\section{Benchmark Results}
\begin{itemize}
\item Accuracy: \textbf{{{accuracy}}}\%
\item Precision: \textbf{{{precision}}}\%
\item Recall: \textbf{{{recall}}}\%
\item F1 Score: \textbf{{{f1_score}}}
\end{itemize}

\section{Conclusion}
This study demonstrates the effectiveness of the \textbf{{{model_name}}} model. Future work will explore hyperparameter tuning and model compression for real-time inference.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""

def format_shape_history_for_latex(shape_history):
    items = ""
    for layer_name, shape in shape_history:
        items += f"\\item {layer_name}: {shape}\n"
    return items


def generate_research_paper(model_data, results, shape_history, shape_prop_img="shape_propagation.png"):
    model_name = model_data["name"]
    title = f"Training and Evaluation of {model_name}"
    author = "Neural Research Team"
    date = datetime.date.today().strftime("%B %d, %Y")
    
    layer_details = "\n".join([f"\\item {layer['type']} ({layer.get('params', {})})" for layer in model_data["layers"]])
    shape_history_items = format_shape_history_for_latex(shape_history)
    
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
        shape_history_items=shape_history_items,
        shape_prop_img=shape_prop_img
    )
    
    filename = f"{model_name}_paper.tex"
    with open(filename, "w") as file:
        file.write(latex_content)
    
    print(f"Research paper saved as {filename}. Compile with LaTeX.")
