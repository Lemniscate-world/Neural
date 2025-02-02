import unittest
from parser import ModelTransformer
from lark import Tree

class TestModelTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = ModelTransformer()

    def test_layer_transformation(self):
        items = [
            Tree('Conv2D', [('filters', '32'), ('kernel_size', '(3, 3)'), ('activation', '"relu"')]),
            Tree('Dense', [('units', '128'), ('activation', '"relu"')]),
            Tree('Output', [('units', '10'), ('activation', '"softmax"')]),
            Tree('Loss', [('value', '"categorical_crossentropy"')]),
            Tree('Optimizer', [('value', '"adam"')]),
        ]
        expected_output = {
            'type': 'Conv2D',
            'params': [('filters', '32'), ('kernel_size', '(3, 3)'), ('activation', '"relu"')],
            'name': 'Conv2D',
            'input': None,
            'layers': None,
            'output': None,
            'loss': {'type': 'Loss', 'value': '"categorical_crossentropy"'},
            'optimizer': {'type': 'Optimizer', 'value': '"adam"'},
        }
        self.assertEqual(self.transformer.layer(items), expected_output)

if __name__ == '__main__':
    unittest.main()