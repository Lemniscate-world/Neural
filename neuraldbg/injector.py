"""neuraldbg.injector — AST-based code injection for zero-code NeuralDBG integration.

Transforms a training script so that NeuralDBG hooks are injected
automatically without the user modifying their source code.

Strategy:
1. Find `model = SomeModel(...)` or `model = AutoModel.from_pretrained(...)`
2. Wrap the training loop body with `with NeuralDbg(model) as dbg:`
3. Inject `dbg.step_iteration()` after `optimizer.step()`
4. Inject `dbg.record_loss(loss.item())` after loss computation
5. Inject `dbg.dump_events()` at the end, saved to a temp JSON file
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import List, Optional

# ── AST Helpers ──────────────────────────────────────────────────────────────


def _is_model_assign(node: ast.stmt) -> Optional[str]:
    """Return model variable name if this stmt is a model instantiation."""
    if not isinstance(node, ast.Assign):
        return None
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    value = node.value
    if isinstance(value, ast.Call):
        # Check if it looks like a model: nn.Sequential(...), AutoModel.from_pretrained(...), MyModel(...)
        func = value.func
        if isinstance(func, ast.Attribute):
            # e.g. nn.Linear, AutoModel.from_pretrained
            import_keywords = {
                "nn",
                "models",
                "AutoModel",
                "AutoModelForCausalLM",
                "transformers",
            }
            if isinstance(func.value, ast.Name) and func.value.id in import_keywords:
                return target.id
            if isinstance(func.value, ast.Attribute):
                return target.id
            return target.id
        elif isinstance(func, ast.Name):
            return target.id
    return None


def _is_optimizer_step(node: ast.stmt) -> bool:
    """Check if statement is `optimizer.step()` or similar."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr == "step"  # optimizer.step(), scheduler.step()
    return False


def _is_loss_assign(node: ast.stmt) -> Optional[str]:
    """Return loss variable name if this stmt assigns to a variable named 'loss'."""
    if not isinstance(node, ast.Assign):
        return None
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    if target.id.lower() != "loss":
        return None
    value = node.value
    if not isinstance(value, ast.Call):
        return None
    return target.id


def _is_training_loop(node: ast.stmt) -> bool:
    """Check if this is a for-loop that looks like a training loop."""
    if not isinstance(node, (ast.For, ast.While)):
        return False
    if isinstance(node, ast.For):
        target_str = ast.unparse(node.target)
        for kw in ("epoch", "step", "iteration", "batch", "i"):
            if kw in target_str.lower():
                return True
    return True  # While loops always wrapped


def _make_import_statement(module: str, names: List[str]) -> ast.ImportFrom:
    """Create `from module import name1, name2`."""
    return ast.ImportFrom(
        module=module,
        names=[ast.alias(name=n, asname=None) for n in names],
        level=0,
    )


def _make_expr(code: str) -> ast.Expr:
    """Parse a string expression into an AST Expr node."""
    return ast.parse(code.strip(), mode="eval").body  # type: ignore


def _make_stmt(code: str) -> ast.stmt:
    """Parse a string statement into an AST stmt node."""
    return ast.parse(code.strip(), mode="exec").body[0]  # type: ignore


def _make_stmts(code: str) -> List[ast.stmt]:
    """Parse multiple statements."""
    tree = ast.parse(code.strip(), mode="exec")
    return list(tree.body)


# ── Main Injector ────────────────────────────────────────────────────────────


def inject_neuraldbg_wrapper(source: str, script_name: str = "training") -> str:
    """Inject NeuralDBG hooks into a training script's source code.

    Args:
        source: Original Python source code.
        script_name: Name of the script (for generated file naming).

    Returns:
        Modified source code with NeuralDBG wrapper injected.
    """
    tree = ast.parse(source)
    model_var: Optional[str] = None
    loss_var: Optional[str] = None

    # Step 1: Find model variable and loss variable
    for node in ast.walk(tree):
        mv = _is_model_assign(node)
        if mv and not model_var:
            model_var = mv
        lv = _is_loss_assign(node)
        if lv and not loss_var:
            loss_var = lv

    # Step 2: Find the training loop and wrap its body
    loop_found = False
    new_body: List[ast.stmt] = []

    for stmt in tree.body:
        if (
            not loop_found
            and _is_training_loop(stmt)
            and isinstance(stmt, (ast.For, ast.While))
        ):
            loop_found = True
            # Wrap the loop body with NeuralDbg context
            wrapped = _wrap_loop_body(stmt, model_var, loss_var)
            new_body.append(wrapped)
        else:
            new_body.append(stmt)

    tree.body = new_body

    # Step 3: Add import for NeuralDbg at the top
    import_stmt = _make_import_statement("neuraldbg", ["NeuralDbg"])
    tree.body.insert(0, import_stmt)

    # Step 4: Add event export at the end
    tree.body.extend(_make_stmts(_EPILOG_TEMPLATE.format(script_name=script_name)))

    # Reconstruct source (add lineno/col_offset for unparse)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _wrap_loop_body(
    loop: ast.stmt,
    model_var: Optional[str],
    loss_var: Optional[str],
) -> ast.stmt:
    """Wrap a training loop's body with `with NeuralDbg(model) as dbg:`.

    The injected code:
    <pre>
    with NeuralDbg(model) as dbg:
        for epoch in ...:
            ...
            loss = criterion(...)
            dbg.record_loss(loss.item())   # injected after loss
            loss.backward()
            optimizer.step()
            dbg.step_iteration()           # injected after optimizer.step()
    </pre>
    """
    body = loop.body if isinstance(loop, (ast.For, ast.While)) else [loop]

    # Inject record_loss after loss computation
    if loss_var:
        injected_body: List[ast.stmt] = []
        for stmt in body:
            injected_body.append(stmt)
            lv = _is_loss_assign(stmt)
            if lv and lv == loss_var:
                injected_body.append(_make_stmt(f"dbg.record_loss({loss_var}.item())"))
        body = injected_body

    # Inject step_iteration after optimizer.step()
    injected_body = []
    for stmt in body:
        injected_body.append(stmt)
        if _is_optimizer_step(stmt):
            injected_body.append(_make_stmt("dbg.step_iteration()"))
    body = injected_body

    # Build the With statement
    model_ref = ast.Name(id=model_var or "model", ctx=ast.Load())
    with_item = ast.withitem(
        context_expr=ast.Call(
            func=ast.Name(id="NeuralDbg", ctx=ast.Load()),
            args=[model_ref],
            keywords=[],
        ),
        optional_vars=ast.Name(id="dbg", ctx=ast.Store()),
    )

    # Create a new For/While with the injected body
    wrapped_loop = copy.deepcopy(loop)
    if hasattr(wrapped_loop, "body"):
        wrapped_loop.body = body  # type: ignore

    with_stmt = ast.With(
        items=[with_item],
        body=[wrapped_loop],
        type_comment=None,
    )
    return with_stmt


# ── Epilog template ──────────────────────────────────────────────────────────

_EPILOG_TEMPLATE = """\

# --- NeuralDBG auto-generated epilog ---
import json as _nd_json
from pathlib import Path as _nd_Path
_nd_events = getattr(dbg, 'dump_events', lambda: [])()
_nd_path = _nd_Path(__file__).parent / "_neuraldbg_{script_name}.events.json" if '__file__' in dir() else _nd_Path("_neuraldbg_{script_name}.events.json")
with open(str(_nd_path), 'w', encoding='utf-8') as _nd_f:
    _nd_json.dump({{"version": "1.3.2", "events": _nd_events}}, _nd_f, default=str)
"""


# ── Test injection ───────────────────────────────────────────────────────────


def test_inject_neuraldbg():
    """Verify injection on a simple training script."""
    sample = """\
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    x = torch.randn(32, 10)
    y = torch.randn(32, 2)
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
"""

    injected = inject_neuraldbg_wrapper(sample, "test_script")
    print("=== Injected Source ===")
    print(injected)

    # Verify key lines exist
    required = (
        "from neuraldbg import NeuralDbg",
        "NeuralDbg(model) as dbg",
        "dbg.record_loss",
        "dbg.step_iteration",
        "dump_events",
    )
    missing = [token for token in required if token not in injected]
    if missing:
        raise SystemExit(f"Injection incomplete, manquant: {missing}")
    print("\\nAll assertions passed!")


if __name__ == "__main__":
    test_inject_neuraldbg()
