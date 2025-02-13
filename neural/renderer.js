class NetworkRenderer {
    constructor(svgId) {
        this.svg = d3.select(`#${svgId}`);
        this.width = +this.svg.attr("width");
        this.height = +this.svg.attr("height");
        this.simulation = null;
    }

    render(data) {
        this.svg.selectAll("*").remove();

        this.simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(this.width / 2, this.height / 2));

        const link = this.svg.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .style("stroke", "#999")
            .style("stroke-width", 2);

        const node = this.svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .call(this.drag(this.simulation));

        node.append("circle")
            .attr("r", 20)
            .style("fill", d => this.getNodeColor(d.type));

        node.append("text")
            .text(d => d.type)
            .attr("dy", 30)
            .attr("text-anchor", "middle");

        this.simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("transform", d => `translate(${d.x},${d.y})`);
        });
    }

    getNodeColor(type) {
        const colors = {
            'Input': '#4CAF50',
            'Conv2D': '#2196F3',
            'MaxPooling2D': '#9C27B0',
            'Dense': '#FF9800',
            'Output': '#795548'
        };
        return colors[type] || '#607D8B';
    }

    drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }
}

// Initialize the renderer
const renderer = new NetworkRenderer('network-svg');

// Example visualization function
function visualize() {
    const codeEditor = document.getElementById('code-editor');
    const code = codeEditor.value;
    
    // For testing, use sample data
    const sampleData = {
        nodes: [
            { id: "input", type: "Input", shape: [28, 28, 1] },
            { id: "layer1", type: "Conv2D", params: { filters: 32, kernel_size: [3, 3] } },
            { id: "layer2", type: "MaxPooling2D", params: { pool_size: [2, 2] } },
            { id: "layer3", type: "Dense", params: { units: 128 } },
            { id: "output", type: "Output", params: { units: 10 } }
        ],
        links: [
            { source: "input", target: "layer1" },
            { source: "layer1", target: "layer2" },
            { source: "layer2", target: "layer3" },
            { source: "layer3", target: "output" }
        ]
    };

    renderer.render(sampleData);
}