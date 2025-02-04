document.addEventListener("DOMContentLoaded", () => {
    const svg = d3.select("svg");
    const nodes = [
        { id: "Input", group: 1 },
        { id: "Conv2D", group: 2 },
        { id: "MaxPooling2D", group: 2 },
        { id: "Dense", group: 3 },
        { id: "Output", group: 4 },
    ];
    const links = [
        { source: "Input", target: "Conv2D" },
        { source: "Conv2D", target: "MaxPooling2D" },
        { source: "MaxPooling2D", target: "Dense" },
        { source: "Dense", target: "Output" },
    ];

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(400, 300));

    const link = svg.selectAll("line")
        .data(links)
        .enter().append("line")
        .style("stroke", "#999");

    const node = svg.selectAll("circle")
        .data(nodes)
        .enter().append("circle")
        .attr("r", 10)
        .style("fill", d => d3.schemeCategory10[d.group]);

    simulation.on("tick", () => {
        node.attr("cx", d => d.x).attr("cy", d => d.y);
        link.attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
    });

    // Add click event to each node

    node.on("click", function(event, d) {
        document.getElementById("node-info").innerHTML = `<h3>${d.id}</h3>`;
    });
    

});
