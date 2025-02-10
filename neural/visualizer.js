d3.json("model.json").then(data => {
    const simulation = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id(d => d.id))
      .force("charge", d3.forceManyBody().strength(-1000))
      .force("center", d3.forceCenter(width/2, height/2));
  });

node.append("title") // Tooltip
  .text(d => `${d.type}\nParams: ${d.params}`);

node.call(d3.drag() // Dragging
  .on("start", dragstarted)
  .on("drag", dragged)
  .on("end", dragended));