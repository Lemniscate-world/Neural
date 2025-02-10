let width = window.innerWidth;
let height = window.innerHeight - 40; // Account for toolbar

const svg = d3.select("#network-container")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

const zoom = d3.zoom()
    .scaleExtent([0.1, 2])
    .on("zoom", (event) => {
        container.attr("transform", event.transform);
    });

svg.call(zoom);

const container = svg.append("g");
const tooltip = d3.select("body").append("div").attr("class", "tooltip");

// Initialize force simulation
const simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2));

// Load model data
d3.json("sample-model.json").then(model => {
    // Create links
    const link = container.append("g")
        .selectAll("line")
        .data(model.links)
        .enter().append("line")
        .attr("class", "link");

    // Create nodes
    const node = container.append("g")
        .selectAll("circle")
        .data(model.nodes)
        .enter().append("circle")
        .attr("class", d => `node ${d.type}`)
        .attr("r", 20)
        .call(drag(simulation))
        .on("mouseover", showTooltip)
        .on("mouseout", hideTooltip);

    // Update simulation
    simulation.nodes(model.nodes);
    simulation.force("link").links(model.links);

    // Tick handler
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    });

    // Physics toggle
    window.togglePhysics = () => {
        simulation.alpha(0.1).restart();
    };

    // Zoom control
    d3.select("#zoom").on("input", function() {
        const scale = d3.select(this).property("value");
        container.attr("transform", `scale(${scale})`);
    });
});

// Drag functions
function drag(simulation) {
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

// Tooltip functions
function showTooltip(event, d) {
    tooltip
        .style("left", `${event.pageX + 15}px`)
        .style("top", `${event.pageY}px`)
        .html(`
            <strong>${d.type}</strong><br>
            ${d.params ? Object.entries(d.params).map(([k,v]) => 
                `${k}: ${Array.isArray(v) ? `[${v.join(', ')}]` : v}`
            ).join('<br>') : ''}
        `)
        .style("opacity", 1);
}

function hideTooltip() {
    tooltip.style("opacity", 0);
}