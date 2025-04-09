import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Link from 'next/link';
import axios from 'axios';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import dynamic from 'next/dynamic';
import { FiCode, FiEye, FiTerminal, FiArrowLeft, FiInfo, FiDownload, FiShare2, FiExternalLink } from 'react-icons/fi';

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
  const [debugInfo, setDebugInfo] = useState<string>('');

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
        if (modelData.annotations) {
          setDebugInfo(JSON.stringify(modelData.annotations, null, 2));

          if (Array.isArray(modelData.annotations.sections)) {
            // New format with sections array inside annotations object
            setCodeSections(modelData.annotations.sections as CodeSection[]);
          } else if (Array.isArray(modelData.annotations)) {
            // Old format with annotations as array
            setCodeSections(modelData.annotations as CodeSection[]);
          } else {
            console.error('Unexpected annotations format:', modelData.annotations);
            setCodeSections([]);
          }
        } else {
          console.error('No annotations found in model data');
          setCodeSections([]);
        }

        // Parse DSL to get visualization data
        try {
          // Add a small delay to ensure the model data is fully loaded
          setTimeout(async () => {
            try {
              console.log('Parsing DSL code:', modelData.dsl_code.substring(0, 100) + '...');
              const response = await axios.post('/api/parse', {
                code: modelData.dsl_code,
                backend: 'tensorflow',
              });

              console.log('Parse response received:', response.data);
              // Create nodes and links for visualization
              createGraphData(response.data.model_data, response.data.shape_history);
            } catch (err) {
              console.error('Failed to parse DSL:', err);
              // Continue without visualization data
            }
          }, 500);
        } catch (err) {
          console.error('Error in DSL parsing setup:', err);
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

  // Parallax scroll effect
  const { scrollY } = useScroll();
  const headerRef = useRef<HTMLDivElement>(null);
  const y = useTransform(scrollY, [0, 300], [0, -50]);
  const opacity = useTransform(scrollY, [0, 100, 200], [1, 0.8, 0]);
  const scale = useTransform(scrollY, [0, 300], [1, 0.9]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-neural-dark text-white flex flex-col items-center justify-center">
        <div className="relative w-24 h-24">
          <div className="absolute top-0 left-0 w-full h-full rounded-full border-4 border-neural-secondary border-opacity-20"></div>
          <div className="absolute top-0 left-0 w-full h-full rounded-full border-t-4 border-neural-secondary animate-spin"></div>
          <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
            <div className="w-12 h-12 bg-neural-dark rounded-full flex items-center justify-center">
              <div className="w-8 h-8 bg-neural-secondary rounded-full opacity-80 animate-pulse"></div>
            </div>
          </div>
        </div>
        <p className="mt-6 text-xl font-light text-neural-secondary animate-pulse">Loading model...</p>
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="min-h-screen bg-neural-dark text-white flex flex-col items-center justify-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-md w-full bg-neural-primary p-8 rounded-xl shadow-2xl border border-gray-800"
        >
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-900 bg-opacity-20 flex items-center justify-center">
            <FiInfo className="text-4xl text-neural-secondary" />
          </div>
          <h1 className="text-2xl font-bold text-neural-secondary mb-4 text-center">Error Loading Model</h1>
          <p className="text-gray-300 mb-8 text-center">{error || 'The requested model could not be found'}</p>
          <button
            className="w-full px-6 py-3 bg-neural-secondary text-white rounded-lg hover:bg-opacity-90 transition-all duration-300 flex items-center justify-center space-x-2 shadow-lg"
            onClick={() => router.push('/models')}
          >
            <FiArrowLeft />
            <span>Back to Models</span>
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neural-dark text-white">
      <Head>
        <title>{model.name} | NeuralPaper.ai</title>
        <meta name="description" content={model.description} />
      </Head>

      {/* Hero section with parallax effect */}
      <motion.div
        ref={headerRef}
        style={{ y, scale }}
        className="relative h-80 bg-gradient-to-br from-neural-primary to-neural-dark overflow-hidden"
      >
        <div className="absolute inset-0 bg-neural-dark opacity-40"></div>
        <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>

        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute rounded-full bg-neural-secondary opacity-10"
              style={{
                width: Math.random() * 100 + 50,
                height: Math.random() * 100 + 50,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, Math.random() * 100 - 50],
                opacity: [0.05, 0.2, 0.05],
              }}
              transition={{
                duration: Math.random() * 10 + 10,
                repeat: Infinity,
                repeatType: "reverse",
              }}
            />
          ))}
        </div>

        {/* Back button */}
        <div className="absolute top-6 left-6 z-10">
          <Link href="/models">
            <button className="flex items-center space-x-2 bg-neural-dark bg-opacity-50 hover:bg-opacity-70 text-white px-4 py-2 rounded-full transition-all duration-300 backdrop-blur-sm">
              <FiArrowLeft />
              <span>Back</span>
            </button>
          </Link>
        </div>

        {/* Share and download buttons */}
        <div className="absolute top-6 right-6 z-10 flex space-x-2">
          <button
            className="bg-neural-dark bg-opacity-50 hover:bg-opacity-70 text-white p-2 rounded-full transition-all duration-300 backdrop-blur-sm"
            onClick={() => navigator.clipboard.writeText(window.location.href)}
            title="Share model"
          >
            <FiShare2 size={18} />
          </button>
          <button
            className="bg-neural-dark bg-opacity-50 hover:bg-opacity-70 text-white p-2 rounded-full transition-all duration-300 backdrop-blur-sm"
            onClick={() => {
              const element = document.createElement("a");
              const file = new Blob([model.dsl_code], {type: 'text/plain'});
              element.href = URL.createObjectURL(file);
              element.download = `${model.id}.neural`;
              document.body.appendChild(element);
              element.click();
              document.body.removeChild(element);
            }}
            title="Download DSL code"
          >
            <FiDownload size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="container mx-auto px-6 h-full flex flex-col justify-end pb-16 relative z-10">
          <motion.div style={{ opacity }}>
            <h1 className="text-5xl font-bold mb-3 text-white">{model.name}</h1>
            <p className="text-xl text-gray-200 max-w-2xl">{model.description}</p>
          </motion.div>
        </div>
      </motion.div>

      <main className="container mx-auto px-4 -mt-10 relative z-10">
        {/* Floating tab navigation */}
        <motion.div
          className="bg-neural-primary rounded-xl shadow-2xl mb-8 overflow-hidden border border-gray-800"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="flex border-b border-gray-700">
            <button
              className={`flex-1 px-6 py-4 flex items-center justify-center space-x-2 transition-all duration-300 ${
                activeTab === 'code'
                  ? 'bg-neural-secondary bg-opacity-20 text-neural-secondary border-b-2 border-neural-secondary'
                  : 'text-gray-400 hover:bg-neural-dark hover:bg-opacity-30 hover:text-white'
              }`}
              onClick={() => setActiveTab('code')}
            >
              <FiCode className={activeTab === 'code' ? 'animate-pulse' : ''} size={18} />
              <span className="font-medium">Annotated Code</span>
            </button>
            <button
              className={`flex-1 px-6 py-4 flex items-center justify-center space-x-2 transition-all duration-300 ${
                activeTab === 'visualization'
                  ? 'bg-neural-secondary bg-opacity-20 text-neural-secondary border-b-2 border-neural-secondary'
                  : 'text-gray-400 hover:bg-neural-dark hover:bg-opacity-30 hover:text-white'
              }`}
              onClick={() => setActiveTab('visualization')}
            >
              <FiEye className={activeTab === 'visualization' ? 'animate-pulse' : ''} size={18} />
              <span className="font-medium">Visualization</span>
            </button>
            <button
              className={`flex-1 px-6 py-4 flex items-center justify-center space-x-2 transition-all duration-300 ${
                activeTab === 'debug'
                  ? 'bg-neural-secondary bg-opacity-20 text-neural-secondary border-b-2 border-neural-secondary'
                  : 'text-gray-400 hover:bg-neural-dark hover:bg-opacity-30 hover:text-white'
              }`}
              onClick={() => setActiveTab('debug')}
            >
              <FiTerminal className={activeTab === 'debug' ? 'animate-pulse' : ''} size={18} />
              <span className="font-medium">Debug</span>
            </button>
          </div>
        </motion.div>

        <div className="content-container mb-16">
          <AnimatePresence mode="wait" initial={false}>
            {activeTab === 'code' && (
              <motion.div
                key="code"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="shadow-2xl rounded-xl overflow-hidden"
              >
                {codeSections.length > 0 ? (
                  <CodeAnnotation
                    code={model.dsl_code}
                    language="yaml"
                    sections={codeSections}
                    title={`${model.name} Neural DSL`}
                  />
                ) : (
                  <div className="bg-neural-primary p-8 rounded-xl border border-gray-800">
                    <div className="flex items-center mb-6">
                      <div className="w-10 h-10 rounded-full bg-neural-secondary bg-opacity-20 flex items-center justify-center mr-4">
                        <FiCode className="text-neural-secondary" />
                      </div>
                      <h3 className="text-2xl font-semibold text-neural-secondary">
                        Neural DSL Code
                      </h3>
                    </div>
                    <pre className="bg-[#1E1E2E] p-6 rounded-lg overflow-auto shadow-inner border border-gray-800">
                      <code className="text-sm">{model.dsl_code}</code>
                    </pre>
                    <div className="mt-6 p-4 bg-neural-dark bg-opacity-50 rounded-lg border border-gray-800">
                      <p className="text-gray-300 italic flex items-center">
                        <FiInfo className="mr-2 text-neural-secondary" />
                        Annotations not available for this model yet.
                      </p>
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'visualization' && (
              <motion.div
                key="visualization"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="bg-neural-primary rounded-xl overflow-hidden shadow-2xl border border-gray-800"
              >
                <div className="p-6 border-b border-gray-800 flex items-center">
                  <div className="w-10 h-10 rounded-full bg-neural-secondary bg-opacity-20 flex items-center justify-center mr-4">
                    <FiEye className="text-neural-secondary" />
                  </div>
                  <h3 className="text-2xl font-semibold text-neural-secondary">
                    Model Architecture
                  </h3>
                </div>

                <div className="h-[700px] bg-neural-dark rounded-lg shadow-inner border border-gray-800 overflow-hidden">
                  {nodes.length > 0 ? (
                    <ModelViewer nodes={nodes} links={links} />
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full">
                      <div className="w-20 h-20 rounded-full bg-neural-primary flex items-center justify-center mb-4 animate-pulse">
                        <FiEye className="text-4xl text-gray-400" />
                      </div>
                      <p className="text-gray-300 text-lg font-medium">Visualization data not available</p>
                      <p className="text-gray-400 text-sm mt-2 max-w-md text-center">The model structure could not be parsed for visualization. This may be due to an unsupported model type or syntax.</p>

                      {debugInfo && (
                        <div className="mt-4 p-4 bg-gray-800 rounded-md w-full max-w-md overflow-auto max-h-[200px] text-xs">
                          <div className="flex items-center mb-2 text-neural-secondary">
                            <FiInfo className="mr-1" />
                            <span>Debug Information</span>
                          </div>
                          <pre className="text-gray-300 whitespace-pre-wrap">
                            {debugInfo}
                          </pre>
                        </div>
                      )}

                      <button
                        className="mt-6 px-4 py-2 bg-neural-secondary text-white rounded-md hover:bg-opacity-90 transition-all"
                        onClick={() => {
                          if (model) {
                            try {
                              console.log('Manually parsing DSL code:', model.dsl_code.substring(0, 100) + '...');
                              axios.post('/api/parse', {
                                code: model.dsl_code,
                                backend: 'tensorflow',
                              }).then(response => {
                                console.log('Parse response received:', response.data);
                                createGraphData(response.data.model_data, response.data.shape_history);
                              }).catch(err => {
                                console.error('Manual parse failed:', err);
                                setDebugInfo(JSON.stringify({
                                  error: err.message,
                                  response: err.response?.data,
                                  status: err.response?.status
                                }, null, 2));
                              });
                            } catch (err) {
                              console.error('Failed to parse DSL:', err);
                            }
                          }
                        }}
                      >
                        Retry Visualization
                      </button>
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {activeTab === 'debug' && (
              <motion.div
                key="debug"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="shadow-2xl rounded-xl overflow-hidden border border-gray-800"
              >
                <DebugPanel modelId={model.id} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Model metadata footer */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="bg-neural-primary p-6 rounded-xl mb-8 shadow-lg border border-gray-800"
        >
          <h3 className="text-xl font-semibold mb-4 text-neural-secondary">Model Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-neural-dark bg-opacity-50 p-4 rounded-lg border border-gray-800">
              <h4 className="text-sm text-gray-400 mb-2">Model ID</h4>
              <p className="text-white font-mono">{model.id}</p>
            </div>
            <div className="bg-neural-dark bg-opacity-50 p-4 rounded-lg border border-gray-800">
              <h4 className="text-sm text-gray-400 mb-2">Annotations</h4>
              <p className="text-white">{codeSections.length} sections</p>
            </div>
            <div className="bg-neural-dark bg-opacity-50 p-4 rounded-lg border border-gray-800">
              <h4 className="text-sm text-gray-400 mb-2">Layers</h4>
              <p className="text-white">{nodes.length > 0 ? nodes.length - 1 : 'Unknown'}</p>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
