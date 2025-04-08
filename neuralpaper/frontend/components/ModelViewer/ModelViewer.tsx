import { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

interface Node {
  id: string;
  type: string;
  x?: number;
  y?: number;
  shape?: number[];
  params?: Record<string, any>;
}

interface Link {
  source: string | Node;
  target: string | Node;
}

interface ModelViewerProps {
  nodes: Node[];
  links: Link[];
  onNodeClick?: (node: Node) => void;
}

const ModelViewer: React.FC<ModelViewerProps> = ({ nodes, links, onNodeClick }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Set up the simulation
    const simulation = d3.forceSimulation<Node>(nodes)
      .force("link", d3.forceLink<Node, Link>(links).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(dimensions.width / 2, dimensions.height / 2));

    // Create the links
    const link = svg.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", 2);

    // Create the nodes
    const node = svg.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(nodes)
      .enter()
      .append("g")
      .call(d3.drag<SVGGElement, Node>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("click", (event, d) => {
        setSelectedNode(d);
        if (onNodeClick) onNodeClick(d);
      });

    // Add circles to the nodes
    node.append("circle")
      .attr("r", d => getNodeSize(d))
      .attr("fill", d => getNodeColor(d.type))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5);

    // Add labels to the nodes
    node.append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .text(d => d.type)
      .attr("fill", "#fff")
      .attr("font-size", "10px");

    // Add shape info below the node
    node.append("text")
      .attr("dy", 18)
      .attr("text-anchor", "middle")
      .text(d => d.shape ? formatShape(d.shape) : "")
      .attr("fill", "#ccc")
      .attr("font-size", "8px");

    // Update positions on each tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => (d.source as Node).x || 0)
        .attr("y1", d => (d.source as Node).y || 0)
        .attr("x2", d => (d.target as Node).x || 0)
        .attr("y2", d => (d.target as Node).y || 0);

      node
        .attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
    });

    // Drag functions
    function dragstarted(event: d3.D3DragEvent<SVGGElement, Node, Node>) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, Node, Node>) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, Node, Node>) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [nodes, links, dimensions, onNodeClick]);

  // Update dimensions on window resize
  useEffect(() => {
    const handleResize = () => {
      if (svgRef.current) {
        const { width, height } = svgRef.current.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Helper functions
  const getNodeSize = (node: Node) => {
    switch (node.type) {
      case 'Input':
        return 25;
      case 'Output':
        return 25;
      default:
        return 20;
    }
  };

  const getNodeColor = (type: string) => {
    const colors: Record<string, string> = {
      'Input': '#4CAF50',
      'Output': '#F44336',
      'Conv2D': '#2196F3',
      'Dense': '#FF9800',
      'Flatten': '#9C27B0',
      'MaxPooling2D': '#00BCD4',
      'Dropout': '#607D8B',
      'BatchNormalization': '#FFEB3B',
      'ResidualConnection': '#E91E63',
    };
    return colors[type] || '#607D8B';
  };

  const formatShape = (shape: number[]) => {
    return shape.join('×');
  };

  return (
    <div className="relative w-full h-full">
      <svg 
        ref={svgRef} 
        className="w-full h-full bg-neural-dark rounded-lg"
        width={dimensions.width}
        height={dimensions.height}
      />
      
      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute bottom-4 right-4 bg-neural-primary p-4 rounded-lg shadow-lg max-w-xs"
        >
          <h3 className="text-lg font-semibold mb-2">{selectedNode.type}</h3>
          {selectedNode.shape && (
            <p className="text-sm text-gray-300 mb-2">
              Shape: {formatShape(selectedNode.shape)}
            </p>
          )}
          {selectedNode.params && Object.keys(selectedNode.params).length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-1">Parameters:</h4>
              <ul className="text-xs text-gray-300">
                {Object.entries(selectedNode.params).map(([key, value]) => (
                  <li key={key}>
                    <span className="font-medium">{key}:</span> {JSON.stringify(value)}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <button 
            className="mt-2 text-xs text-neural-secondary hover:underline"
            onClick={() => setSelectedNode(null)}
          >
            Close
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default ModelViewer;
