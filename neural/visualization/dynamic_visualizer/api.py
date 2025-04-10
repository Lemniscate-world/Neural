from flask import Flask, request, jsonify
from flask_cors import CORS
from parser.parser import create_parser, ModelTransformer
from neural.visualization.static_visualizer.visualizer import NeuralVisualizer

app = Flask(__name__)
CORS(app, resources={r"/parse": {"origins": "*"}})

@app.route('/parse', methods=['POST'])
def parse_network():
    try:
        # Get raw text data instead of JSON
        code = request.data.decode('utf-8')
        print("Received code:", code)
        parser = create_parser('network')
        tree = parser.parse(code)
        print(tree.pretty())
        model_data = ModelTransformer().transform(tree)
        print(model_data)
        
        visualizer = NeuralVisualizer(model_data)
        visualization_data = visualizer.model_to_d3_json()
        print(visualization_data)
        
        return jsonify(visualization_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
