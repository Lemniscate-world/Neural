import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import axios from 'axios';
import { motion } from 'framer-motion';
import dynamic from 'next/dynamic';

// Import components
import CodeAnnotation from '../../components/CodeAnnotation/CodeAnnotation';
import ModelViewer from '../../components/ModelViewer/ModelViewer';

// Import DebugPanel dynamically to avoid SSR issues with Plotly
const DebugPanel = dynamic(() => import('../../components/DebugPanel/DebugPanel'), { ssr: false });

// Model interfaces

interface ModelData {
  id: string;
  name: string;
  description: string;
  dsl_code: string;
  annotations: Record<string, CodeSection>;
}

interface CodeSection {
  id: string;
  code: string;
  annotation: string;
  lineStart: number;
  lineEnd: number;
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

export default function ModelPage() {
  const router = useRouter();
  const { id } = router.query;

  const [model, setModel] = useState<ModelData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'code' | 'visualization' | 'debug'>('code');
  const [nodes, setNodes] = useState<Node[]>([]);
  const [links, setLinks] = useState<Link[]>([]);
  const [codeSections, setCodeSections] = useState<CodeSection[]>([]);

  // Fetch model data
  useEffect(() => {
    if (!id) return;

    const fetchModel = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Fetch model data from the API
        const response = await axios.get(`/api/models/${id}`);
        const modelData = response.data;

        if (!modelData) {
          throw new Error('Model not found');
        }

        setModel(modelData);

        // Parse annotations into sections
        if (Array.isArray(modelData.annotations)) {
          setCodeSections(modelData.annotations as CodeSection[]);
        } else {
          setCodeSections([]);
        }

        // Parse DSL to get visualization data
        try {
          const response = await axios.post('/api/parse', {
            code: modelData.dsl_code,
            backend: 'tensorflow',
          });

          // Create nodes and links for visualization
          createGraphData(response.data.model_data, response.data.shape_history);
        } catch (err) {
          console.error('Failed to parse DSL:', err);
          // Continue without visualization data
        }

        setIsLoading(false);
      } catch (err: any) {
        setError(err.message || 'Failed to load model');
        setIsLoading(false);
      }
    };

    fetchModel();
  }, [id]);

  // Create graph data for visualization
  const createGraphData = (modelData: any, shapeHistory: any[]) => {
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

  if (isLoading) {
    return (
      <div className="min-h-screen bg-neural-dark text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-neural-secondary"></div>
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="min-h-screen bg-neural-dark text-white flex flex-col items-center justify-center">
        <h1 className="text-2xl font-bold text-neural-secondary mb-4">Error</h1>
        <p className="text-gray-300 mb-6">{error || 'Model not found'}</p>
        <button
          className="px-4 py-2 bg-neural-secondary text-white rounded hover:bg-opacity-90"
          onClick={() => router.push('/models')}
        >
          Back to Models
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neural-dark text-white">
      <Head>
        <title>{model.name} | NeuralPaper.ai</title>
        <meta name="description" content={model.description} />
      </Head>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 text-neural-secondary">{model.name}</h1>
          <p className="text-xl text-gray-300">{model.description}</p>
        </div>

        <div className="mb-6">
          <div className="flex border-b border-gray-700">
            <button
              className={`px-4 py-2 ${
                activeTab === 'code'
                  ? 'border-b-2 border-neural-secondary text-neural-secondary'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('code')}
            >
              Annotated Code
            </button>
            <button
              className={`px-4 py-2 ${
                activeTab === 'visualization'
                  ? 'border-b-2 border-neural-secondary text-neural-secondary'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('visualization')}
            >
              Visualization
            </button>
            <button
              className={`px-4 py-2 ${
                activeTab === 'debug'
                  ? 'border-b-2 border-neural-secondary text-neural-secondary'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('debug')}
            >
              Debug
            </button>
          </div>
        </div>

        <div className="content-container">
          {activeTab === 'code' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              {codeSections.length > 0 ? (
                <CodeAnnotation
                  code={model.dsl_code}
                  language="yaml"
                  sections={codeSections}
                  title={`${model.name} Neural DSL`}
                />
              ) : (
                <div className="bg-neural-primary p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-4 text-neural-secondary">
                    Neural DSL Code
                  </h3>
                  <pre className="bg-[#1E1E2E] p-4 rounded overflow-auto">
                    <code>{model.dsl_code}</code>
                  </pre>
                  <p className="mt-4 text-gray-400 italic">
                    Annotations not available for this model yet.
                  </p>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'visualization' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="bg-neural-primary rounded-lg overflow-hidden"
            >
              <div className="p-4 border-b border-gray-700">
                <h3 className="text-xl font-semibold text-neural-secondary">
                  Model Architecture
                </h3>
              </div>

              <div className="h-[600px]">
                {nodes.length > 0 ? (
                  <ModelViewer nodes={nodes} links={links} />
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    Visualization data not available
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {activeTab === 'debug' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <DebugPanel modelId={model.id} />
            </motion.div>
          )}
        </div>
      </main>
    </div>
  );
}
