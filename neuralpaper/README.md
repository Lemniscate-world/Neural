# NeuralPaper.ai

<p align="center">
  <img src="../docs/images/neuralpaper_architecture.png" alt="NeuralPaper Architecture" width="600"/>
</p>

NeuralPaper.ai is a platform that integrates Neural DSL with NeuralDbg to create interactive, annotated neural network models with visualization features similar to [nn.labml.ai](https://nn.labml.ai). It provides a web-based interface for creating, visualizing, and sharing neural network models with detailed explanations and interactive elements.

## Features

- **Annotated Models**: Explore neural network architectures with detailed annotations explaining each component
- **Interactive Visualization**: Visualize model architecture, shape propagation, and computation flow in real-time
- **Live Debugging**: Debug models with NeuralDbg to analyze gradients, activations, and performance
- **DSL Playground**: Experiment with Neural DSL to create and modify models with instant feedback
- **Educational Resources**: Learn about neural network concepts with interactive examples and tutorials

## Project Structure

```
neuralpaper/
├── backend/
│   ├── api/                  # FastAPI server for Neural DSL execution
│   ├── models/               # Annotated model implementations
│   └── integrations/         # Connectors to Neural and NeuralDbg
├── frontend/
│   ├── components/           # React components
│   │   ├── ModelViewer/      # Interactive model visualization
│   │   ├── CodeAnnotation/   # Side-by-side code and annotations
│   │   ├── DSLPlayground/    # Interactive DSL editor
│   │   └── DebugPanel/       # NeuralDbg integration
│   ├── pages/                # Next.js pages
│   │   ├── models/           # Model showcase pages
│   │   ├── playground/       # DSL playground page
│   │   └── blog/             # Blog section
│   └── public/               # Static assets
└── shared/                   # Shared types and utilities
```

## Getting Started

### Prerequisites

- Node.js 14+ and npm
- Python 3.8+
- Neural DSL and NeuralDbg installed

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neuralpaper.git
   cd neuralpaper
   ```

2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. Initialize the database with sample models:
   ```bash
   cd backend/scripts
   python init_db.py
   ```

2. Start the backend server:
   ```bash
   cd backend
   uvicorn api.main:app --reload
   ```

3. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:3000`

Alternatively, you can use the provided `start.sh` script to start both the backend and frontend servers:
```bash
cd neuralpaper
./start.sh
```

## Adding New Models

To add a new annotated model:

1. Create a Neural DSL file in `backend/models/` (e.g., `mymodel.neural`)
2. Create an annotations file in `backend/models/` (e.g., `mymodel.annotations.json`) with the following structure:
   ```json
   {
     "name": "My Model",
     "description": "Description of my model",
     "category": "Computer Vision",
     "complexity": "Medium",
     "sections": [
       {
         "id": "section1",
         "lineStart": 1,
         "lineEnd": 5,
         "annotation": "Explanation of this section"
       }
     ]
   }
   ```
3. The model will be automatically available in the frontend

## Making the Implementation Fully Real

The current implementation includes some mock data and placeholders. To make it fully real:

1. **Complete the WebSocket Integration**:
   - Enhance the WebSocket endpoints in `backend/api/main.py` to stream real data from NeuralDbg
   - Update the `neural_connector.py` to capture and forward NeuralDbg output

2. **Implement Real-Time Debugging**:
   - Add methods to query the running NeuralDbg instance for real-time data
   - Replace the mock trace data with real data from your running NeuralDbg instance

3. **Add Database Integration**:
   - Replace the file-based storage with a proper database (SQLite, PostgreSQL, etc.)
   - Implement proper model management with CRUD operations

4. **Enhance the Frontend**:
   - Add more interactive visualizations
   - Implement user authentication
   - Add a blog section for tutorials and articles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Neural DSL](https://github.com/Lemniscate-SHA-256/Neural) - The domain-specific language for neural networks
- [NeuralDbg](https://github.com/Lemniscate-SHA-256/Neural) - The neural network debugger
- [nn.labml.ai](https://nn.labml.ai) - Inspiration for the annotated model format
