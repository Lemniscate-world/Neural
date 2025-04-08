import React, { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import axios from 'axios';
import WebSocketClient from './WebSocketClient';

// Import Plotly dynamically to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface TraceEntry {
  layer: string;
  input_shape: number[];
  output_shape: number[];
  flops: number;
  memory: number;
  execution_time: number;
  compute_time: number;
  transfer_time: number;
  grad_norm?: number;
  dead_ratio?: number;
  mean_activation?: number;
  anomaly?: boolean;
}

interface DebugPanelProps {
  sessionId?: string;
  modelId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const DebugPanel: React.FC<DebugPanelProps> = ({
  sessionId,
  modelId,
  autoRefresh = true,
  refreshInterval = 2000,
}) => {
  const [traceData, setTraceData] = useState<TraceEntry[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'execution' | 'gradients' | 'memory' | 'anomalies'>('execution');

  // Fetch trace data
  const fetchTraceData = async () => {
    if (!sessionId && !modelId) return;

    setIsLoading(true);
    setError(null);

    try {
      // Fetch trace data from the API
      const endpoint = sessionId ? `/api/debug/${sessionId}/trace` : `/api/models/${modelId}/trace`;
      const response = await axios.get(endpoint);

      setTraceData(response.data || []);
      setIsLoading(false);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch trace data');
      setIsLoading(false);
    }
  };

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data: any) => {
    if (Array.isArray(data)) {
      setTraceData(data);
      setIsLoading(false);
    }
  }, []);

  // Auto-refresh data
  useEffect(() => {
    // Initial fetch
    fetchTraceData();

    // Only use interval if WebSocket is not available or not connected
    if (autoRefresh && !sessionId) {
      const interval = setInterval(fetchTraceData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [modelId, autoRefresh, refreshInterval, sessionId]);

  // Create execution time chart
  const renderExecutionTimeChart = () => {
    if (traceData.length === 0) return null;

    const layers = traceData.map(entry => entry.layer);
    const executionTimes = traceData.map(entry => entry.execution_time * 1000); // Convert to ms
    const computeTimes = traceData.map(entry => entry.compute_time * 1000);
    const transferTimes = traceData.map(entry => entry.transfer_time * 1000);

    return (
      <Plot
        data={[
          {
            x: layers,
            y: executionTimes,
            type: 'bar',
            name: 'Total Execution Time (ms)',
            marker: { color: '#e94560' }
          },
          {
            x: layers,
            y: computeTimes,
            type: 'bar',
            name: 'Compute Time (ms)',
            marker: { color: '#0ea5e9' }
          },
          {
            x: layers,
            y: transferTimes,
            type: 'bar',
            name: 'Transfer Time (ms)',
            marker: { color: '#10b981' }
          }
        ]}
        layout={{
          title: 'Layer Execution Time',
          xaxis: { title: 'Layer' },
          yaxis: { title: 'Time (ms)' },
          barmode: 'group',
          plot_bgcolor: '#1a1a2e',
          paper_bgcolor: '#1a1a2e',
          font: { color: '#ffffff' },
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
        style={{ width: '100%', height: '400px' }}
      />
    );
  };

  // Create gradient flow chart
  const renderGradientFlowChart = () => {
    if (traceData.length === 0) return null;

    const layers = traceData.map(entry => entry.layer);
    const gradNorms = traceData.map(entry => entry.grad_norm || 0);

    return (
      <Plot
        data={[
          {
            x: layers,
            y: gradNorms,
            type: 'bar',
            name: 'Gradient Magnitude',
            marker: { color: '#0ea5e9' }
          }
        ]}
        layout={{
          title: 'Gradient Flow',
          xaxis: { title: 'Layer' },
          yaxis: { title: 'Gradient Magnitude' },
          plot_bgcolor: '#1a1a2e',
          paper_bgcolor: '#1a1a2e',
          font: { color: '#ffffff' },
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
        style={{ width: '100%', height: '400px' }}
      />
    );
  };

  // Create memory usage chart
  const renderMemoryUsageChart = () => {
    if (traceData.length === 0) return null;

    const layers = traceData.map(entry => entry.layer);
    const memory = traceData.map(entry => entry.memory / (1024 * 1024)); // Convert to MB
    const flops = traceData.map(entry => entry.flops / 1000000); // Convert to MFLOPs

    return (
      <Plot
        data={[
          {
            x: layers,
            y: memory,
            type: 'bar',
            name: 'Memory Usage (MB)',
            marker: { color: '#0ea5e9' },
            yaxis: 'y'
          },
          {
            x: layers,
            y: flops,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'FLOPs (M)',
            marker: { color: '#10b981' },
            yaxis: 'y2'
          }
        ]}
        layout={{
          title: 'Memory Usage & Computational Complexity',
          xaxis: { title: 'Layer' },
          yaxis: { title: 'Memory (MB)', side: 'left' },
          yaxis2: {
            title: 'FLOPs (M)',
            side: 'right',
            overlaying: 'y',
            showgrid: false
          },
          legend: { x: 0, y: 1.2 },
          plot_bgcolor: '#1a1a2e',
          paper_bgcolor: '#1a1a2e',
          font: { color: '#ffffff' },
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
        style={{ width: '100%', height: '400px' }}
      />
    );
  };

  // Create anomalies chart
  const renderAnomaliesChart = () => {
    if (traceData.length === 0) return null;

    const layers = traceData.map(entry => entry.layer);
    const activations = traceData.map(entry => entry.mean_activation || 0);
    const deadRatios = traceData.map(entry => (entry.dead_ratio || 0) * 100); // Convert to percentage
    const anomalies = traceData.map(entry => entry.anomaly ? 1 : 0);

    return (
      <Plot
        data={[
          {
            x: layers,
            y: activations,
            type: 'bar',
            name: 'Mean Activation',
            marker: { color: '#0ea5e9' },
            yaxis: 'y'
          },
          {
            x: layers,
            y: deadRatios,
            type: 'bar',
            name: 'Dead Neurons (%)',
            marker: { color: '#e94560' },
            yaxis: 'y2'
          },
          {
            x: layers,
            y: anomalies,
            type: 'scatter',
            mode: 'markers',
            name: 'Anomaly Detected',
            marker: {
              color: '#f59e0b',
              size: 12,
              symbol: 'triangle-up'
            }
          }
        ]}
        layout={{
          title: 'Activation Anomalies & Dead Neurons',
          xaxis: { title: 'Layer' },
          yaxis: { title: 'Activation Magnitude', side: 'left' },
          yaxis2: {
            title: 'Dead Neurons (%)',
            side: 'right',
            overlaying: 'y',
            range: [0, 100],
            showgrid: false
          },
          legend: { x: 0, y: 1.2 },
          plot_bgcolor: '#1a1a2e',
          paper_bgcolor: '#1a1a2e',
          font: { color: '#ffffff' },
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
        style={{ width: '100%', height: '400px' }}
      />
    );
  };

  // Determine WebSocket URL if session ID is available
  const wsUrl = sessionId ? `ws://localhost:8000/ws/debug/${sessionId}` : null;

  return (
    <div className="debug-panel bg-neural-dark rounded-lg overflow-hidden">
      {/* WebSocket client for real-time updates */}
      {wsUrl && (
        <WebSocketClient
          url={wsUrl}
          onMessage={handleWebSocketMessage}
          onConnect={() => console.log('Connected to NeuralDbg WebSocket')}
          onDisconnect={() => console.log('Disconnected from NeuralDbg WebSocket')}
          onError={(e) => console.error('WebSocket error:', e)}
        />
      )}

      <div className="bg-neural-primary p-3 flex justify-between items-center">
        <h3 className="text-lg font-semibold text-white">
          NeuralDbg Dashboard {wsUrl && <span className="text-green-400 text-xs ml-2">(Live)</span>}
        </h3>

        <div className="flex space-x-2">
          <button
            className="bg-neural-secondary text-white px-3 py-1 rounded text-sm hover:bg-opacity-90"
            onClick={fetchTraceData}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>

          <a
            href="http://localhost:8050"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-opacity-90"
          >
            Open Full Dashboard
          </a>
        </div>
      </div>

      <div className="tab-navigation bg-neural-primary border-t border-gray-700 p-2 flex space-x-2">
        <button
          className={`px-3 py-1 rounded text-sm ${
            activeTab === 'execution'
              ? 'bg-neural-secondary text-white'
              : 'bg-neural-dark text-gray-300 hover:bg-opacity-90'
          }`}
          onClick={() => setActiveTab('execution')}
        >
          Execution Time
        </button>
        <button
          className={`px-3 py-1 rounded text-sm ${
            activeTab === 'gradients'
              ? 'bg-neural-secondary text-white'
              : 'bg-neural-dark text-gray-300 hover:bg-opacity-90'
          }`}
          onClick={() => setActiveTab('gradients')}
        >
          Gradient Flow
        </button>
        <button
          className={`px-3 py-1 rounded text-sm ${
            activeTab === 'memory'
              ? 'bg-neural-secondary text-white'
              : 'bg-neural-dark text-gray-300 hover:bg-opacity-90'
          }`}
          onClick={() => setActiveTab('memory')}
        >
          Memory & FLOPs
        </button>
        <button
          className={`px-3 py-1 rounded text-sm ${
            activeTab === 'anomalies'
              ? 'bg-neural-secondary text-white'
              : 'bg-neural-dark text-gray-300 hover:bg-opacity-90'
          }`}
          onClick={() => setActiveTab('anomalies')}
        >
          Anomalies
        </button>
      </div>

      <div className="chart-container p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-96">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-neural-secondary"></div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-96 text-red-400">
            {error}
          </div>
        ) : traceData.length === 0 ? (
          <div className="flex items-center justify-center h-96 text-gray-400">
            No trace data available
          </div>
        ) : (
          <>
            {activeTab === 'execution' && renderExecutionTimeChart()}
            {activeTab === 'gradients' && renderGradientFlowChart()}
            {activeTab === 'memory' && renderMemoryUsageChart()}
            {activeTab === 'anomalies' && renderAnomaliesChart()}
          </>
        )}
      </div>

      {traceData.length > 0 && (
        <div className="layer-details bg-neural-primary p-4 border-t border-gray-700">
          <h4 className="text-md font-semibold mb-2 text-neural-secondary">Layer Details</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-700">
              <thead>
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Layer</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Shape</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Time (ms)</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Memory (MB)</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">FLOPs (M)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {traceData.map((entry, index) => (
                  <tr key={index} className={entry.anomaly ? 'bg-red-900 bg-opacity-20' : ''}>
                    <td className="px-4 py-2 text-sm">{entry.layer}</td>
                    <td className="px-4 py-2 text-sm">{entry.output_shape.join('×')}</td>
                    <td className="px-4 py-2 text-sm">{(entry.execution_time * 1000).toFixed(2)}</td>
                    <td className="px-4 py-2 text-sm">{(entry.memory / (1024 * 1024)).toFixed(2)}</td>
                    <td className="px-4 py-2 text-sm">{(entry.flops / 1000000).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default DebugPanel;
