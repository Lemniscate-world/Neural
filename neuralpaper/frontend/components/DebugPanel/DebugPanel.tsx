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

      <div className="bg-neural-primary p-4 flex justify-between items-center border-b border-gray-700">
        <div className="flex items-center">
          <div className="w-8 h-8 rounded-full bg-neural-secondary bg-opacity-20 flex items-center justify-center mr-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-neural-secondary" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 5a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2h-2.22l.123.489.804.804A1 1 0 0113 18H7a1 1 0 01-.707-1.707l.804-.804L7.22 15H5a2 2 0 01-2-2V5zm5.771 7H5V5h10v7H8.771z" clipRule="evenodd" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-white">
            NeuralDbg Dashboard {wsUrl && <span className="text-green-400 text-xs ml-2 px-1.5 py-0.5 bg-green-900 bg-opacity-30 rounded-md">(Live)</span>}
          </h3>
        </div>

        <div className="flex space-x-2">
          <button
            className="bg-neural-secondary text-white px-3 py-1.5 rounded-md text-sm font-medium hover:bg-opacity-90 transition-colors shadow-sm flex items-center"
            onClick={fetchTraceData}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Loading...
              </>
            ) : 'Refresh'}
          </button>

          <a
            href="http://localhost:8050"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-blue-600 text-white px-3 py-1.5 rounded-md text-sm font-medium hover:bg-opacity-90 transition-colors shadow-sm flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            Open Full Dashboard
          </a>
        </div>
      </div>

      <div className="tab-navigation bg-neural-primary border-b border-gray-700 p-3 flex space-x-3 overflow-x-auto">
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center ${
            activeTab === 'execution'
              ? 'bg-neural-secondary text-white shadow-md'
              : 'bg-neural-dark bg-opacity-50 text-gray-300 hover:bg-opacity-70 hover:text-white'
          }`}
          onClick={() => setActiveTab('execution')}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Execution Time
        </button>
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center ${
            activeTab === 'gradients'
              ? 'bg-neural-secondary text-white shadow-md'
              : 'bg-neural-dark bg-opacity-50 text-gray-300 hover:bg-opacity-70 hover:text-white'
          }`}
          onClick={() => setActiveTab('gradients')}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
          Gradient Flow
        </button>
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center ${
            activeTab === 'memory'
              ? 'bg-neural-secondary text-white shadow-md'
              : 'bg-neural-dark bg-opacity-50 text-gray-300 hover:bg-opacity-70 hover:text-white'
          }`}
          onClick={() => setActiveTab('memory')}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Memory & FLOPs
        </button>
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center ${
            activeTab === 'anomalies'
              ? 'bg-neural-secondary text-white shadow-md'
              : 'bg-neural-dark bg-opacity-50 text-gray-300 hover:bg-opacity-70 hover:text-white'
          }`}
          onClick={() => setActiveTab('anomalies')}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          Anomalies
        </button>
      </div>

      <div className="chart-container p-6 bg-neural-dark">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-96 bg-neural-dark bg-opacity-50 rounded-lg border border-gray-800 shadow-inner">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-neural-secondary mb-4"></div>
            <p className="text-gray-400 text-sm">Loading trace data...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-96 bg-red-900 bg-opacity-10 rounded-lg border border-red-800 shadow-inner">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-red-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-red-400 font-medium">{error}</p>
            <button
              className="mt-4 px-4 py-2 bg-red-800 bg-opacity-50 hover:bg-opacity-70 text-white rounded-md text-sm transition-colors"
              onClick={fetchTraceData}
            >
              Try Again
            </button>
          </div>
        ) : traceData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-96 bg-neural-dark bg-opacity-50 rounded-lg border border-gray-800 shadow-inner">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-gray-300 font-medium">No trace data available</p>
            <p className="text-gray-400 text-sm mt-2 max-w-md text-center">Start a debug session to see execution traces and performance metrics.</p>
            <button
              className="mt-4 px-4 py-2 bg-neural-secondary text-white rounded-md text-sm hover:bg-opacity-90 transition-colors"
              onClick={fetchTraceData}
            >
              Refresh Data
            </button>
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
        <div className="layer-details bg-neural-primary p-5 border-t border-gray-700">
          <div className="flex items-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-neural-secondary mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <h4 className="text-lg font-semibold text-neural-secondary">Layer Details</h4>
          </div>
          <div className="overflow-x-auto bg-neural-dark bg-opacity-50 rounded-lg border border-gray-800 shadow-inner">
            <table className="min-w-full divide-y divide-gray-700">
              <thead>
                <tr className="bg-neural-dark bg-opacity-70">
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Layer</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Shape</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Time (ms)</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Memory (MB)</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">FLOPs (M)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {traceData.map((entry, index) => (
                  <tr key={index} className={`${index % 2 === 0 ? 'bg-neural-dark bg-opacity-30' : ''} ${entry.anomaly ? 'bg-red-900 bg-opacity-20' : ''} hover:bg-neural-primary hover:bg-opacity-30 transition-colors`}>
                    <td className="px-4 py-3 text-sm font-medium">{entry.layer}</td>
                    <td className="px-4 py-3 text-sm font-mono text-gray-300">{entry.output_shape.join('×')}</td>
                    <td className="px-4 py-3 text-sm">{(entry.execution_time * 1000).toFixed(2)}</td>
                    <td className="px-4 py-3 text-sm">{(entry.memory / (1024 * 1024)).toFixed(2)}</td>
                    <td className="px-4 py-3 text-sm">{(entry.flops / 1000000).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 text-xs text-gray-400 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Rows highlighted in red indicate potential anomalies in layer execution</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default DebugPanel;
