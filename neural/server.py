from flask import Flask, jsonify, request
from parser import create_parser, ModelTransformer
from lark import Lark
from visualizer import NeuralVisualizer

app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse_model():
    # Get .nr/.neural file content from request
    nr_content = request.data.decode('utf-8')
    
    # Parse using your existing code
    parser = create_parser()
    parsed = parser.parse(nr_content)
    model_data = ModelTransformer().transform(parsed)
    
    # Convert to D3 format
    d3_data = NeuralVisualizer(model_data).model_to_d3_json(model_data)
    
    print(jsonify(d3_data))
    return jsonify(d3_data)

if __name__ == '__main__':
    app.run(port=5000)