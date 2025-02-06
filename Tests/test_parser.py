from lark import Lark
import pytest

def test_create_parser_returns_lark():
    parser = create_parser('network')
    assert isinstance(parser, Lark)

def test_network_parser_valid():
    network_text = (
        "network MyNet {"
        "  input: (28, 28)"
        "  layers:\n"
        "      Dense(units=128)\n"
        "      Output(units=10, activation=\"softmax\")\n"
        "  loss: \"mse\"\n"
        "  optimizer: \"sgd\"\n"
        "}"
    )
    tree = network_parser.parse(network_text)
    # Ensure the parse tree is not empty
    assert tree is not None

def test_layer_parser_valid():
    # A simple layer definition using the 'Dense' rule.
    layer_text = "Dense(units=128)"
    tree = layer_parser.parse(layer_text)
    assert tree is not None

def test_research_parser_valid():
    research_text = (
        "research MyResearch {"
        "  metrics { accuracy: 0.95 }"
        "  references { paper:\"paper1\" paper:\"paper2\" }"
        "}"
    )
    tree = research_parser.parse(research_text)
    assert tree is not None

def test_network_parser_invalid():
    # Missing network name and required sections should cause a parse error.
    invalid_text = "network { input: (28,28) layers: Dense(units=128) loss: \"mse\" optimizer: \"sgd\" }"
    with pytest.raises(Exception):
        network_parser.parse(invalid_text)