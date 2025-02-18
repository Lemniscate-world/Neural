class NetworkRenderer {
    constructor(svgId) {
        this.svg = d3.select(`#${svgId}`);
        this.width = +this.svg.attr("width");
        this.height = +this.svg.attr("height");
        this.simulation = null;
    }

    async parseAndVisualize(code) {
        try {
            const response = await fetch('http://localhost:5000/parse', {
                method: 'POST',
                headers: {
                    'Content-Type': 'text/plain',
                },
                body: code
            });

            if (!response.ok) {
                throw new Error('Parser API request failed');
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            this.render(data);
        } catch (error) {
            console.error('Error:', error);
            this.showError(error.message);
        }
    }

    showError(message) {
        const errorDiv = document.getElementById('error-message') || 
            document.createElement('div');
        errorDiv.id = 'error-message';
        errorDiv.style.color = 'red';
        errorDiv.textContent = message;
        this.svg.node().parentNode.appendChild(errorDiv);
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

// Update visualization function
function visualize() {
    const codeEditor = document.getElementById('code-editor');
    const code = codeEditor.value;
    renderer.parseAndVisualize(code);
}