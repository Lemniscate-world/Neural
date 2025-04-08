import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import axios from 'axios';
import { motion } from 'framer-motion';
import ModelViewer from '../ModelViewer/ModelViewer';

interface ShapeHistoryItem {
  layer_id: string;
  layer_type: string;
  input_shape: number[];
  output_shape: number[];
}

interface Node {
  id: string;
  type: string;
  shape?: number[];
  params?: Record<string, any>;
}

interface Link {
  source: string;
  target: string;
}

interface DSLPlaygroundProps {
  initialCode?: string;
  height?: string;
}

const DEFAULT_CODE = `network SimpleConvNet {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Conv2D(filters=64, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
}`;

const DSLPlayground: React.FC<DSLPlaygroundProps> = ({
  initialCode = DEFAULT_CODE,
  height = '600px',
}) => {
  const [code, setCode] = useState(initialCode);
  const [modelData, setModelData] = useState<any>(null);
  const [shapeHistory, setShapeHistory] = useState<ShapeHistoryItem[]>([]);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [links, setLinks] = useState<Link[]>([]);
  const [selectedBackend, setSelectedBackend] = useState<string>('tensorflow');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'visualization' | 'code'>('visualization');

  // Parse DSL code when it changes
  const parseDSL = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/parse', {
        code,
        backend: selectedBackend,
      });
      
      setModelData(response.data.model_data);
      setShapeHistory(response.data.shape_history);
      
      // Create nodes and links for visualization
      createGraphData(response.data.model_data, response.data.shape_history);
      
      setIsLoading(false);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to parse DSL code');
      setIsLoading(false);
    }
  };

  // Generate code from DSL
  const generateCode = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/generate', {
        code,
        backend: selectedBackend,
      });
      
      setGeneratedCode(response.data.code);
      setActiveTab('code');
      setIsLoading(false);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate code');
      setIsLoading(false);
    }
  };

  // Create graph data for visualization
  const createGraphData = (modelData: any, shapeHistory: ShapeHistoryItem[]) => {
    if (!modelData || !shapeHistory) return;
    
    const newNodes: Node[] = [];
    const newLinks: Link[] = [];
    
    // Add input node
    newNodes.push({
      id: 'input',
      type: 'Input',
      shape: modelData.input.shape,
    });
    
    // Add layer nodes
    modelData.layers.forEach((layer: any, index: number) => {
      const nodeId = `layer_${index}`;
      const shapeInfo = shapeHistory.find(item => item.layer_id === nodeId);
      
      newNodes.push({
        id: nodeId,
        type: layer.type,
        shape: shapeInfo?.output_shape,
        params: layer.params,
      });
      
      // Connect to previous node
      const sourceId = index === 0 ? 'input' : `layer_${index - 1}`;
      newLinks.push({
        source: sourceId,
        target: nodeId,
      });
    });
    
    // Add output node if not already included
    if (modelData.layers.length > 0) {
      const lastLayerId = `layer_${modelData.layers.length - 1}`;
      const lastLayer = modelData.layers[modelData.layers.length - 1];
      
      // Only add output node if the last layer is not already an output
      if (lastLayer.type !== 'Output') {
        const outputId = 'output';
        const lastShapeInfo = shapeHistory[shapeHistory.length - 1];
        
        newNodes.push({
          id: outputId,
          type: 'Output',
          shape: lastShapeInfo?.output_shape,
        });
        
        newLinks.push({
          source: lastLayerId,
          target: outputId,
        });
      }
    }
    
    setNodes(newNodes);
    setLinks(newLinks);
  };

  // Handle code changes
  const handleEditorChange = (value: string | undefined) => {
    if (value !== undefined) {
      setCode(value);
    }
  };

  // Start debug session
  const startDebugSession = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/debug', {
        code,
        backend: selectedBackend,
      });
      
      // Open debug dashboard in new window
      window.open(response.data.dashboard_url, '_blank');
      
      setIsLoading(false);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start debug session');
      setIsLoading(false);
    }
  };

  return (
    <div className="dsl-playground bg-neural-dark rounded-lg overflow-hidden">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
        {/* Editor Panel */}
        <div className="editor-panel flex flex-col">
          <div className="bg-neural-primary p-3 flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Neural DSL Editor</h3>
            <div className="flex space-x-2">
              <select
                className="bg-neural-dark text-white px-2 py-1 rounded text-sm"
                value={selectedBackend}
                onChange={(e) => setSelectedBackend(e.target.value)}
              >
                <option value="tensorflow">TensorFlow</option>
                <option value="pytorch">PyTorch</option>
              </select>
              <button
                className="bg-neural-secondary text-white px-3 py-1 rounded text-sm hover:bg-opacity-90"
                onClick={parseDSL}
                disabled={isLoading}
              >
                {isLoading ? 'Processing...' : 'Run'}
              </button>
            </div>
          </div>
          
          <div className="flex-grow" style={{ height }}>
            <Editor
              height="100%"
              defaultLanguage="yaml"
              defaultValue={code}
              onChange={handleEditorChange}
              theme="vs-dark"
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                wordWrap: 'on',
                scrollBeyondLastLine: false,
                lineNumbers: 'on',
                glyphMargin: false,
                folding: true,
                lineDecorationsWidth: 10,
                automaticLayout: true,
              }}
            />
          </div>
          
          <div className="bg-neural-primary p-3 flex justify-between items-center">
            <div className="flex space-x-2">
              <button
                className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-opacity-90"
                onClick={generateCode}
                disabled={isLoading}
              >
                Generate Code
              </button>
              <button
                className="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-opacity-90"
                onClick={startDebugSession}
                disabled={isLoading}
              >
                Debug
              </button>
            </div>
            
            {error && (
              <div className="text-red-400 text-sm">
                {error}
              </div>
            )}
          </div>
        </div>
        
        {/* Visualization Panel */}
        <div className="visualization-panel flex flex-col">
          <div className="bg-neural-primary p-3 flex justify-between items-center">
            <div className="flex space-x-2">
              <button
                className={`px-3 py-1 rounded text-sm ${
                  activeTab === 'visualization'
                    ? 'bg-neural-secondary text-white'
                    : 'bg-neural-dark text-gray-300 hover:bg-opacity-90'
                }`}
                onClick={() => setActiveTab('visualization')}
              >
                Visualization
              </button>
              <button
                className={`px-3 py-1 rounded text-sm ${
                  activeTab === 'code'
                    ? 'bg-neural-secondary text-white'
                    : 'bg-neural-dark text-gray-300 hover:bg-opacity-90'
                }`}
                onClick={() => setActiveTab('code')}
                disabled={!generatedCode}
              >
                Generated Code
              </button>
            </div>
            
            {shapeHistory.length > 0 && (
              <div className="text-gray-300 text-sm">
                {`${nodes.length} layers`}
              </div>
            )}
          </div>
          
          <div className="flex-grow bg-[#1E1E2E]" style={{ height }}>
            {isLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-neural-secondary"></div>
              </div>
            ) : activeTab === 'visualization' ? (
              nodes.length > 0 ? (
                <ModelViewer nodes={nodes} links={links} />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400">
                  Click "Run" to visualize the model
                </div>
              )
            ) : (
              <div className="h-full overflow-auto">
                {generatedCode ? (
                  <Editor
                    height="100%"
                    defaultLanguage={selectedBackend === 'tensorflow' ? 'python' : 'python'}
                    value={generatedCode}
                    theme="vs-dark"
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      fontSize: 14,
                      wordWrap: 'on',
                      scrollBeyondLastLine: false,
                    }}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    Click "Generate Code" to see the output
                  </div>
                )}
              </div>
            )}
          </div>
          
          {activeTab === 'visualization' && shapeHistory.length > 0 && (
            <div className="bg-neural-primary p-3 overflow-x-auto">
              <div className="flex space-x-4">
                {shapeHistory.map((item, index) => (
                  <div key={index} className="flex-shrink-0 text-xs">
                    <div className="font-medium text-neural-secondary">{item.layer_type}</div>
                    <div className="text-gray-400">
                      {`${item.input_shape.join('×')} → ${item.output_shape.join('×')}`}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DSLPlayground;
