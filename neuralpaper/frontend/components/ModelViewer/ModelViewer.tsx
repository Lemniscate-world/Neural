import { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';
import { FiMaximize2, FiMinimize2, FiZoomIn, FiZoomOut, FiX, FiInfo } from 'react-icons/fi';

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
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);

  // Generate a color scheme based on node types
  const nodeTypes = useMemo(() => {
    return [...new Set(nodes.map(node => node.type))];
  }, [nodes]);

  // Create a color scale
  const colorScale = useMemo(() => {
    return d3.scaleOrdinal<string>()
      .domain(nodeTypes)
      .range([
        '#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0',
        '#00BCD4', '#607D8B', '#FFEB3B', '#E91E63', '#3F51B5',
        '#009688', '#FFC107', '#795548', '#CDDC39', '#673AB7'
      ]);
  }, [nodeTypes]);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Create a zoom behavior
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        setZoom(event.transform.k);
        svg.select('g.zoom-container').attr('transform', event.transform.toString());
      });

    // Apply zoom behavior to SVG
    svg.call(zoomBehavior);

    // Create a container for all elements that should be zoomed
    const container = svg.append('g')
      .attr('class', 'zoom-container');

    // Add a subtle grid pattern
    container.append('rect')
      .attr('width', dimensions.width * 2) // Make grid larger than viewport
      .attr('height', dimensions.height * 2)
      .attr('x', -dimensions.width / 2) // Center the grid
      .attr('y', -dimensions.height / 2)
      .attr('fill', 'none')
      .attr('class', 'bg-grid-pattern');

    // Set up the simulation with more realistic forces
    const simulation = d3.forceSimulation<Node>(nodes)
      .force("link", d3.forceLink<Node, Link>(links).id(d => d.id).distance(150))
      .force("charge", d3.forceManyBody().strength(-800)) // Stronger repulsion
      .force("center", d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
      .force("x", d3.forceX(dimensions.width / 2).strength(0.1)) // Stronger centering
      .force("y", d3.forceY(dimensions.height / 2).strength(0.1)) // Stronger centering
      .force("collision", d3.forceCollide().radius(d => getNodeSize(d) * 2)); // Larger collision radius

    // Create the links with gradient and animation
    const link = container.append("g")
      .attr("class", "links")
      .selectAll("path")
      .data(links)
      .enter()
      .append("path")
      .attr("class", "node-link")
      .attr("stroke", "url(#link-gradient)")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", 2)
      .attr("fill", "none");

    // Add gradient definition for links
    const defs = svg.append("defs");

    const gradient = defs.append("linearGradient")
      .attr("id", "link-gradient")
      .attr("gradientUnits", "userSpaceOnUse");

    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#0f3460");

    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#e94560");

    // Create the nodes with enhanced styling
    const node = container.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .call(d3.drag<SVGGElement, Node>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("click", (event, d) => {
        setSelectedNode(d);
        if (onNodeClick) onNodeClick(d);
      })
      .on("mouseover", (event, d) => {
        setHoveredNode(d);
        d3.select(event.currentTarget).raise();
      })
      .on("mouseout", () => {
        setHoveredNode(null);
      });

    // Add node backgrounds with glow effect
    node.append("circle")
      .attr("r", d => getNodeSize(d) + 3)
      .attr("fill", "rgba(0,0,0,0.3)")
      .attr("class", "node-shadow");

    // Add circles to the nodes
    node.append("circle")
      .attr("r", d => getNodeSize(d))
      .attr("fill", d => colorScale(d.type))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .attr("class", "node-circle")
      .attr("filter", d => d.type === 'Input' || d.type === 'Output' ? 'url(#glow)' : '');

    // Add glow filter
    const filter = defs.append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");

    filter.append("feGaussianBlur")
      .attr("stdDeviation", "3")
      .attr("result", "coloredBlur");

    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Add labels to the nodes
    node.append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .text(d => d.type)
      .attr("fill", "#fff")
      .attr("font-size", "12px")
      .attr("font-weight", "bold")
      .attr("pointer-events", "none");

    // Add shape info below the node
    node.append("text")
      .attr("dy", 20)
      .attr("text-anchor", "middle")
      .text(d => d.shape ? formatShape(d.shape) : "")
      .attr("fill", "#ccc")
      .attr("font-size", "10px")
      .attr("pointer-events", "none");

    // Update positions on each tick with curved links
    simulation.on("tick", () => {
      // Update links as curved paths
      link.attr("d", (d) => {
        const sourceX = (d.source as Node).x || 0;
        const sourceY = (d.source as Node).y || 0;
        const targetX = (d.target as Node).x || 0;
        const targetY = (d.target as Node).y || 0;

        // Calculate control point for curve
        const dx = targetX - sourceX;
        const dy = targetY - sourceY;
        const dr = Math.sqrt(dx * dx + dy * dy) * 1.5; // Curve factor

        // Straight line for short distances, curved for longer ones
        if (dr < 150) {
          return `M${sourceX},${sourceY}L${targetX},${targetY}`;
        } else {
          return `M${sourceX},${sourceY}A${dr},${dr} 0 0,1 ${targetX},${targetY}`;
        }
      });

      // Update node positions with boundary constraints
      node.attr("transform", d => {
        // Keep nodes within boundaries with more padding
        d.x = Math.max(80, Math.min(dimensions.width - 80, d.x || 0));
        d.y = Math.max(80, Math.min(dimensions.height - 80, d.y || 0));
        return `translate(${d.x},${d.y})`;
      });

      // Update gradient positions for links
      links.forEach((link, i) => {
        const sourceX = (link.source as Node).x || 0;
        const sourceY = (link.source as Node).y || 0;
        const targetX = (link.target as Node).x || 0;
        const targetY = (link.target as Node).y || 0;

        gradient.attr("x1", sourceX)
          .attr("y1", sourceY)
          .attr("x2", targetX)
          .attr("y2", targetY);
      });
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

  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    setTimeout(() => {
      handleResize();
    }, 100);
  };

  // Zoom controls
  const handleZoomIn = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>().on('zoom', null);
    svg.transition().call(zoom.scaleBy, 1.3);
  };

  const handleZoomOut = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>().on('zoom', null);
    svg.transition().call(zoom.scaleBy, 0.7);
  };

  const handleResize = () => {
    if (containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect();
      setDimensions({ width, height });
    }
  };

  // Helper functions
  const getNodeSize = (node: Node) => {
    switch (node.type) {
      case 'Input':
        return 30;
      case 'Output':
        return 30;
      case 'Conv2D':
      case 'Dense':
        return 25;
      default:
        return 20;
    }
  };

  const formatShape = (shape: number[]) => {
    return shape.join('×');
  };

  // Update dimensions on resize
  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isFullscreen]);

  return (
    <div
      ref={containerRef}
      className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-neural-dark' : 'w-full h-full'}`}
    >
      {/* Control panel */}
      <div className="absolute top-4 right-4 z-10 flex space-x-2">
        <button
          className="bg-neural-primary bg-opacity-70 hover:bg-opacity-100 p-2 rounded-full transition-all duration-300"
          onClick={handleZoomIn}
          title="Zoom in"
        >
          <FiZoomIn className="text-white" />
        </button>
        <button
          className="bg-neural-primary bg-opacity-70 hover:bg-opacity-100 p-2 rounded-full transition-all duration-300"
          onClick={handleZoomOut}
          title="Zoom out"
        >
          <FiZoomOut className="text-white" />
        </button>
        <button
          className="bg-neural-primary bg-opacity-70 hover:bg-opacity-100 p-2 rounded-full transition-all duration-300"
          onClick={toggleFullscreen}
          title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        >
          {isFullscreen ? <FiMinimize2 className="text-white" /> : <FiMaximize2 className="text-white" />}
        </button>
      </div>

      {/* Legend */}
      <div className="absolute top-4 left-4 z-10 bg-neural-primary bg-opacity-80 p-3 rounded-lg shadow-lg border border-gray-700">
        <h4 className="text-xs font-semibold mb-2 text-white">Layer Types</h4>
        <div className="flex flex-col space-y-1 max-h-[200px] overflow-y-auto pr-2">
          {nodeTypes.map(type => (
            <div key={type} className="flex items-center text-xs">
              <div
                className="w-3 h-3 rounded-full mr-2 flex-shrink-0"
                style={{ backgroundColor: colorScale(type) }}
              />
              <span className="text-white truncate">{type}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Zoom indicator */}
      <div className="absolute bottom-4 left-4 z-10 bg-neural-primary bg-opacity-80 px-3 py-1.5 rounded-md shadow-md border border-gray-700 text-xs font-medium text-white">
        {Math.round(zoom * 100)}%
      </div>

      {/* SVG container */}
      <svg
        ref={svgRef}
        className="w-full h-full bg-neural-dark rounded-lg shadow-inner border border-gray-800"
        width={dimensions.width}
        height={dimensions.height}
      />

      {/* Node hover tooltip */}
      <AnimatePresence>
        {hoveredNode && !selectedNode && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            transition={{ duration: 0.2 }}
            className="absolute z-20 bg-neural-primary bg-opacity-95 p-3 rounded-lg shadow-xl border border-gray-700 max-w-xs pointer-events-none"
            style={{
              left: Math.min(Math.max((hoveredNode.x || 0), 100), dimensions.width - 100),
              top: Math.max((hoveredNode.y || 0) - 80, 20),
              transform: 'translateX(-50%)'
            }}
          >
            <h3 className="text-sm font-semibold text-white">{hoveredNode.type}</h3>
            {hoveredNode.shape && (
              <p className="text-xs text-gray-300">
                Shape: {formatShape(hoveredNode.shape)}
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Selected node details panel */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3, type: 'spring' }}
            className="absolute bottom-4 right-4 bg-neural-primary p-5 rounded-lg shadow-xl max-w-sm border border-gray-700 z-20"
          >
            <div className="flex justify-between items-start mb-3">
              <div className="flex items-center">
                <div
                  className="w-8 h-8 rounded-full mr-3 flex items-center justify-center"
                  style={{ backgroundColor: colorScale(selectedNode.type) }}
                >
                  <FiInfo className="text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white">{selectedNode.type}</h3>
              </div>
              <button
                className="text-gray-400 hover:text-white transition-colors"
                onClick={() => setSelectedNode(null)}
              >
                <FiX />
              </button>
            </div>

            {selectedNode.shape && (
              <div className="mb-4 bg-neural-dark bg-opacity-50 p-3 rounded-lg">
                <h4 className="text-sm font-semibold mb-1 text-gray-300">Shape</h4>
                <p className="text-sm text-white font-mono">
                  {formatShape(selectedNode.shape)}
                </p>
              </div>
            )}

            {selectedNode.params && Object.keys(selectedNode.params).length > 0 && (
              <div>
                <h4 className="text-sm font-semibold mb-2 text-gray-300">Parameters</h4>
                <div className="bg-neural-dark bg-opacity-50 p-3 rounded-lg">
                  <ul className="text-sm text-gray-200 space-y-1">
                    {Object.entries(selectedNode.params).map(([key, value]) => (
                      <li key={key} className="flex justify-between">
                        <span className="font-medium text-neural-secondary">{key}:</span>
                        <span className="font-mono">{JSON.stringify(value)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Fullscreen overlay close button */}
      {isFullscreen && (
        <button
          className="absolute top-4 left-4 z-50 bg-neural-secondary p-2 rounded-full shadow-lg"
          onClick={toggleFullscreen}
        >
          <FiX className="text-white" />
        </button>
      )}
    </div>
  );
};

export default ModelViewer;
