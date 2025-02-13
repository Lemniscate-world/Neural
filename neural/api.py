from flask import Flask, request, jsonify
from flask_cors import CORS
from parser import create_parser, ModelTransformer
from visualizer import NeuralVisualizer

app = Flask(__name__)
CORS(app)

@app.route('/parse', methods=['POST'])
def parse_network():
    try:
        code = request.json['code']
        parser = create_parser('network')
        tree = parser.parse(code)
        model_data = ModelTransformer().transform(tree)
        
        visualizer = NeuralVisualizer(model_data)
        visualization_data = visualizer.model_to_d3_json()
        
        return jsonify(visualization_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)