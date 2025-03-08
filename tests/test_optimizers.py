from neural.automatic_hyperparameter_optimization import hpo 
from neural.parser import parser


# test_optimizers.py
def test_adam_optimizer():
    dsl = """
    network Test {
        optimizer: Adam(learning_rate=0.001, beta_1=0.9)
    }
    """
    parsed = layer_parser.parse(dsl)
    assert parsed["optimizer"] == {
        "type": "Adam",
        "params": {"learning_rate": 0.001, "beta_1": 0.9}
    }

def test_learning_rate_schedule():
    dsl = """
    optimizer: SGD(
        learning_rate=ExponentialDecay(0.1, 1000, 0.96),
        momentum=0.9
    )
    """
    parsed = layer_parser.parse(dsl)
    assert parsed["optimizer"]["params"]["learning_rate"] == {
        "type": "ExponentialDecay",
        "args": [0.1, 1000, 0.96]
    }