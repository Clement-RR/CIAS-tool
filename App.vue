<template>
  <!-- Scenario selection dropdown -->
  <!--div class="dropdown-container">
    <label for="scenario-selector">Complexity:</label>
    <select id="scenario-selector" v-model="selectedScenario" @change="updateMatrices">
      <option value="Simple">Simple</option>
      <option value="Standard">Standard</option>
      <option value="Complex">Complex</option>
    </select>
  </div-->


  <!-- Dropdown for selecting change class -->
  <div class="dropdown-container">
    <label for="change-class-selector">Select Change Class:</label>
    <select id="change-class-selector" v-model="selectedChangeClass" @change="initializeMatrices();updateGraph()">
      <option v-for="(value, key) in changeClasses" :key="key" :value="key">
        {{ key }}
      </option>
    </select>
  </div>

  <!-- Dropdown for selecting start node -->
<div class="dropdown-container">
  <label for="node-selector">Start Node:</label>
  <select id="node-selector" v-model="selectedNode" @change="updateGraph">
    <option v-for="node in nodeList" :key="node.id" :value="node.id">
      {{ node.name }}
    </option>
  </select>
</div>

<!-- Dropdown for selecting start node -->
<div class="dropdown-container">
  <label for="depth-selector">Depth:</label>
  <select id="depth-selector" v-model="selectedDepth" @change="updateGraph()">
    <option value='1'>1</option>
      <option value='2'>2</option>
      <option value='3'>3</option>
      <option value='4'>4</option>
      <option value='5'>5</option>
  </select>
</div>

  <!-- Button container for matrix type selection -->
  <div class="button-container">
    <button :class="{ 'matrix-button': true, 'active': currentMatrix === 'transitionMatrix' }" @click="initializeCurrentMatrix(0); updateGraph()">Transition Probabilities</button>
    <button :class="{ 'matrix-button': true, 'active': currentMatrix === 'mostProbableCostMatrix' }" @click="initializeCurrentMatrix(2); updateGraph()">Costs</button>
    <button :class="{ 'matrix-button': true, 'active': currentMatrix === 'mostProbableTimeMatrix' }" @click="initializeCurrentMatrix(5); updateGraph()">Time</button>
  </div>

  <!-- Network graph container -->
  <div ref="networkContainer" style="width: fit-content; height: 500px;"></div>

  <!-- Run Simulation Button -->
  <div class="run-button">
    <button :class="'button'" @click="sendSimulationData">Run Simulation</button>
  </div>

  <!-- Display graphs -->
  <div>
    <h2>Simulation Results</h2>
    <div v-for="(path, index) in graphPaths" :key="index">
      <img :src="path" :alt="`Graph ${index + 1}`">
    </div>
  </div>
<!-- Reset Button -->
<button @click="resetGraph">Reset</button>

<div class="editors">
  <!-- Matrix value editor modal -->
  <div v-if="showEditModal" class="edit-modal">
    <h3>Edit Matrix Value</h3>
    <p>{{ nodes.get(selectedEdge.from).label }} to {{ nodes.get(selectedEdge.to).label }} transition probability: </p>
    <input type="number" v-model="EdgeValue" step="0.1" min="0" max="1">
    <button @click="applyEdit">Apply</button>
    <button @click="cancelEdit">Cancel</button>
  </div>
  
  <!--Add Node-->
  <div v-if="showAddNode" class="add-node">
    <h3>Add Node</h3>
    <p>From {{ activeNode.label }} to :</p>
    <input type="text" v-model="toNode">
    <p>Existing nodes :</p>
    <select id="existingNode-selector" v-model="toNode">
     <option v-for="node in nodeList" :key="node.name" :value="node.name">
      {{ node.name }}
     </option>
    </select>
    <p> Transition probability :</p>
    <input type="number" v-model="EdgeValue" step="0.1" min="0" max="1">
    <button @click="addNode">Add Node</button>
    <button @click="cancel">Cancel</button>
  </div>
</div>
  

</template>

<script>
import { Network, DataSet } from 'vis-network/standalone';
import { scenarios, changeClasses } from "@/ChangeClassesData";
import axios from 'axios';

export default {
  name: 'App',
  data() {
    return {
      nodes: new DataSet([]),
      edges: new DataSet([]),
      changeClasses: changeClasses,
      selectedScenario: 'Standard',
      selectedChangeClass: 'All',
      scenarioData: scenarios['Standard'],
      currentMatrix: null,  // dynamically set
      graphPaths: [],
      showEditModal: false,
      showAddNode: false,
      selectedEdge: null,
      activeNode: null,
      toNode: null,
      EdgeValue: 0,
      CostValue: 0,
      TimeValue: 0,
      selectedNode: 1, // Default to the first node
      nodeList: [], // Will populate based on the matrix
      selectedDepth: 2,
      tonodeid: 0,
      fullMatrix: {},
      rows:[],
      cols: [],
    };
  },
  created() {
    this.initializeMatrices();
    this.initializeCurrentMatrix(0);
    this.updateMatrices(); // Ensure this is called to set up initial state
    //this.updateGraph();
  },
  methods: {

    initializeMatrices() {
      const scenarioData = this.scenarioData;
      const changeClassData = this.changeClasses[this.selectedChangeClass][this.selectedScenario];

      if (!scenarioData || !scenarioData.matrices) {
        console.error("Matrix data is missing for the selected scenario");
        console.log(this.scenarioData)
        return;
      }

      // Initialize an object to hold all filtered matrices to be sent
      

      // Iterate over each matrix type defined in the scenarioData
      Object.keys(scenarioData.matrices).forEach(matrixType => {
        this.fullMatrix[matrixType] = scenarioData.matrices[matrixType];
        
        
        this.cols = changeClassData.cols;
        this.rows = changeClassData.rows;
        
      });



    },

    initializeCurrentMatrix(i) { //Using matrix based on selected scenario
      const scenarioData = scenarios[this.selectedScenario];
      if (scenarioData && scenarioData.matrices) {
        const availableMatrices = Object.keys(scenarioData.matrices);
        if (availableMatrices.length > 0) {
          this.currentMatrix = availableMatrices[i];  // Set to the matrix type
        }
      }
    },
    


    getNodeNames() {
      
      return this.scenarioData.nodeNames;
      },

    getNodeID(label) {
      const nodeNames = this.getNodeNames();
      return (nodeNames.indexOf(label.replace(/\n/g, " "))+1);
    },

    getNodeLabel(nodeId) {
    const nodeNames = this.getNodeNames();
    const nodeName = nodeNames[nodeId - 1] || `Node ${nodeId}`;
    return this.formatNodeLabel(nodeName);
  },

    formatNodeLabel(label) {
    return label.replace(/(.{1,15})(\s|$)/g, "$1\n").trim();
  },

    updateGraph() {
      
  const classData = this.changeClasses[this.selectedChangeClass][this.selectedScenario];
  if (!this.scenarioData || !this.scenarioData.matrices || !classData.propagationPaths) {
    console.error("Scenario data, matrices, or propagation paths are undefined");
    return;
  }

  const matrix = this.fullMatrix[this.currentMatrix];
  const paths = classData.propagationPaths;
  this.nodes.clear();
  this.edges.clear();

  let queue = [this.selectedNode];
  let visited = 0;
  let tonodeid = 1; //Initializing for first node
  let depth = this.selectedDepth;
  let nodesAdded = 1; //Intitializing for first iteration

  this.nodes.add({ //Adding the first node/SelectedNode
        id: 1,
        label: this.getNodeLabel(this.selectedNode), // Use the formatted label
        color: '#ffc645' // Orange '#ffc645' if selected, otherwise Cornflower Blue '#41e197' 
      });
   
  for (let step = 1; step < depth; step++) {  //for loop for the depth

    let iterations = nodesAdded; //number of nodes added in previous depth level, which need to be iterated through
    nodesAdded = 0;

    for (let step2 = 0; step2<iterations; step2++) { //for loop for each node of each depth level
     visited = visited + 1;
     let currentNode = queue.shift();

     if (!paths[currentNode]) {
      console.warn(`No propagation paths defined for Node ${currentNode}`);
      continue;
     };


     paths[currentNode].forEach(toNode => {
      if (toNode !== this.selectedNode ) {
        queue.push(toNode);
        tonodeid = tonodeid + 1; //Next toNode
        nodesAdded = nodesAdded + 1; // Counting how many nodes are added at each depth level
        
      
      this.nodes.add({ //Adding each toNode
        id: tonodeid,
        label: this.getNodeLabel(toNode), // Use the formatted label
        color: '#41e197' // Orange '#ffc645' if selected, otherwise Cornflower Blue '#41e197' 
      });
      const value = matrix[currentNode - 1][toNode - 1]; // Adjust index to 0-based
      if (value > 0 && toNode !== this.selectedNode) { // Check to prevent creating edges to the start node
        const edgeId = `${visited}-${tonodeid}`;
        
          this.edges.add({
            id: edgeId,
            from: visited,
            to: tonodeid,
            label: `${value}`,
            originalLabel: `${value}`,  // Store the original value when the edge is first created
            arrows: 'to'
          });
        
      }
    }
     });
     }
    }
    this.tonodeid = tonodeid  //Updating the global tonodeid variable
    },

    resetGraph() {
        this.edges.forEach(edge => {
            this.edges.update({
                id: edge.id,
                label: edge.originalLabel,  // Resetting to original label
                hidden: false // Ensure all edges are visible again
            });
        });
    },

    updateMatrices() {
  // Reset node list to ensure fresh data
  this.nodeList = [];

  // Assume the matrix size defines the number of nodes correctly
  const scenarioData = scenarios[this.selectedScenario];
  if (scenarioData.matrixSize) {
    // Populate nodeList for all possible nodes in the scenario
    for (let i = 1; i <= scenarioData.matrixSize; i++) {
      this.nodeList.push({ id: i, name: scenarioData.nodeNames[i - 1] });
    }
  }
  this.updateGraph(); // Call to redraw the graph
    },

    sendSimulationData() {
      let matricesToSend = {};
      Object.keys(this.scenarioData.matrices).forEach(matrixType => {
        
        let rows = this.rows;
        let cols = this.cols;
        let fullMatrix = this.fullMatrix;
        // Filter the matrix based on the rows and columns
        const filteredMatrix = rows.map(row => {
          return cols.map(col => fullMatrix[matrixType][row][col]);
        });

        // Zero out the column corresponding to the selected node
        const selectedNodeIndex = this.selectedNode-1 ;  // Adjusting for 0-based index
        if (cols.includes(selectedNodeIndex)) {
          const columnIndex = cols.indexOf(selectedNodeIndex);
          filteredMatrix.forEach(row => row[columnIndex] = 0);
        };
        
        // Add the filtered matrix to the matricesToSend object with the appropriate key
        matricesToSend[matrixType] = filteredMatrix;
      });
    // Log the matrices to be sent to the console
    console.log("Filtered Matrices being sent:", matricesToSend);

    // Send the filtered matrices to the backend
    axios.post('http://localhost:5000/run_simulation', matricesToSend, {
    headers: {
      'Content-Type': 'application/json'
        },
      })
      .then(response => {
          // Handle response and display graphs
          this.graphPaths = response.data.graphs.map(graph => `http://localhost:5000/${graph}`);
        })
        .catch(error => {
          console.error('Error sending data:', error);
        });
      },

    applyEdit() {
      // Check if there's a selected edge and a valid edit value
        if (this.selectedEdge && (this.EdgeValue) !== undefined) {
          if (this.EdgeValue == 0 )  {
            this.edges.update({
              id: this.selectedEdge.id,
              hidden: true // Hide the edge instead of setting label to 0
            });
            this.nodes.update({
              id: this.nodes.get(this.selectedEdge.to).id,
              hidden: true,
            })
            this.changeClasses[this.selectedChangeClass][this.selectedScenario].propagationPaths[this.getNodeID(this.nodes.get(this.selectedEdge.from).label)].splice(this.changeClasses[this.selectedChangeClass][this.selectedScenario].propagationPaths[this.getNodeID(this.nodes.get(this.selectedEdge.from).label)].indexOf(this.getNodeID(this.nodes.get(this.selectedEdge.to).label)),1)
          } 
    this.fullMatrix['transitionMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] = Number(this.EdgeValue);
      this.fullMatrix['bestCostMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] = Number(this.EdgeValue * 10000);
      this.fullMatrix['mostProbableCostMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] = Number(this.EdgeValue * 10000 * 2);
      this.fullMatrix['worstCostMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] = Number(this.EdgeValue * 10000 * 5);
      this.fullMatrix['bestTimeMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] = Number(this.EdgeValue * 100);
      this.fullMatrix['mostProbableTimeMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] = Number(this.EdgeValue * 100 * 2);
      this.fullMatrix['worstTimeMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] = Number(this.EdgeValue * 100 * 5);
      // Update the edge with the new label
      this.edges.update({
              id: this.selectedEdge.id,
              label: `${this.fullMatrix[this.currentMatrix][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1] }`,
              arrows: 'to',
              hidden: false // Ensure the edge is visible
      });
          this.showEditModal = false;
        } else {
          console.error("No edge selected or invalid value.");
        }
        },

    cancelEdit() {
      this.showEditModal = false;
    },

    addNode() {
      if (this.activeNode && this.toNode !== undefined && this.EdgeValue!== 0 ) {
        //Add Node to all systems if it is a new node
      if (this.nodeList.find(({ name }) => name === this.toNode) == undefined) {
        this.nodeList.push({ id: this.nodeList.length + 1, name: this.toNode }); 
        this.scenarioData.nodeNames.push(this.toNode); //Add new node name to name list
        this.cols.push(this.nodeList.length-1); //Add column index of the new node to the columns to be sent for the simulation
        if (this.selectedDepth>3){ //Only add the row if the depth is large enough
          this.rows.push(this.nodeList.length-1);
        };
        Object.keys(this.scenarioData.matrices).forEach(matrixType => {
          this.fullMatrix[matrixType].forEach(row => {
          row.push(0);
          });
          this.fullMatrix[matrixType].push(new Array(this.nodeList.length).fill(0))
        });
        this.changeClasses[this.selectedChangeClass][this.selectedScenario].propagationPaths[this.getNodeID(this.toNode)] = [];
        
      };
      //Add propagation path array for to node if inexistant
      if (this.changeClasses[this.selectedChangeClass][this.selectedScenario].propagationPaths[this.getNodeID(this.toNode)] == undefined){
        this.changeClasses[this.selectedChangeClass][this.selectedScenario].propagationPaths[this.getNodeID(this.toNode)] = [];
      };


      //Editing the matrix value in each matrix (0-based index)
      this.fullMatrix['transitionMatrix'][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1] = Number(this.EdgeValue);
      this.fullMatrix['bestCostMatrix'][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1] = Number(this.EdgeValue * 10000);
      this.fullMatrix['mostProbableCostMatrix'][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1] = Number(this.EdgeValue * 10000 * 2);
      this.fullMatrix['worstCostMatrix'][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1] = Number(this.EdgeValue * 10000 * 5);
      this.fullMatrix['bestTimeMatrix'][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1] = Number(this.EdgeValue * 100);
      this.fullMatrix['mostProbableTimeMatrix'][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1] = Number(this.EdgeValue * 100 * 2);
      this.fullMatrix['worstTimeMatrix'][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1] = Number(this.EdgeValue * 100 * 5);
      //Adding the representative propagation path to the propagation paths
      this.changeClasses[this.selectedChangeClass][this.selectedScenario].propagationPaths[this.getNodeID(this.activeNode.label)].push(this.getNodeID(this.toNode));
      //Add Node and Edge
      let tonodeid = this.tonodeid + 1
        this.nodes.add({ //Adding each toNode
         id: tonodeid,
         label: this.formatNodeLabel(this.toNode),
         color: '#41e197' // Orange '#ffc645' if selected, otherwise Cornflower Blue '#41e197' 
      });
      
      const edgeId = `${this.activeNode.id}-${tonodeid}`;

      this.edges.add({
        id: edgeId,
        from: this.activeNode.id,
        to: tonodeid,
        label: `${this.fullMatrix[this.currentMatrix][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1]}`,
        originalLabel: `${this.fullMatrix[this.currentMatrix][this.getNodeID(this.activeNode.label)-1][this.getNodeID(this.toNode)-1]}`,  // Store the original value when the edge is first created
        arrows: 'to'
      });
      this.tonodeid = tonodeid;        
      };
      },
      
    },

    cancel() {
      this.showAddNode = false;
    },

  
  mounted() {
    this.updateGraph();
    const container = this.$refs.networkContainer;
    const data = {nodes: this.nodes, edges: this.edges};
    const options = {
      nodes: {
        shape: 'circle',
        size: 16,
        font: {
          size: 14,
          color: '#000000'
        },
        borderWidth: 2
      },
      edges: {
        width: 2,
        color: {color: '#848484', highlight: '#848484', hover: '#848484', inherit: false},
        arrows: {
          to: {enabled: true, scaleFactor: 1, type: 'arrow'}
        },
        font: {
          align: 'top'
        }
      },
      interaction: {
        dragNodes: true
      },
      physics: {
        enabled: true,
        barnesHut: {
          gravitationalConstant: -30000,
            centralGravity: 3,
            springLength: 20,
            springConstant: 0.05,
            damping: 0.3,
            avoidOverlap: 0.5
        }
      }
    };
    this.network = new Network(container, data, options);

    // Correct event registration for edge selection
    this.network.on("selectEdge", (params) => {
        if (params.edges.length) {
            const edgeId = params.edges[0];
            const edgeData = this.edges.get(edgeId);
            this.selectedEdge = edgeData;
            this.EdgeValue = this.fullMatrix['transitionMatrix'][this.getNodeID(this.nodes.get(this.selectedEdge.from).label)-1][this.getNodeID(this.nodes.get(this.selectedEdge.to).label)-1];  // Load the current label for editing
            this.showEditModal = true;  // Show modal to edit value
        }
    });

    this.network.on("selectNode", (params) => {
        if (params.nodes.length) {
            const NodeId = params.nodes[0];
            const NodeData = this.nodes.get(NodeId);
            this.activeNode = NodeData;  // Load the current label for editing
            this.showAddNode = true;  // Show modal to edit value
        }
    });
  },

  watch: {
  selectedNode(newVal, oldVal) {
    if (newVal !== oldVal) {
      this.updateGraph();  // Redraw the graph with the new node styles, ensuring the selected node is highlighted
    }
  },
  nodeList(newList) {
    if (!newList.map(node => node.id).includes(this.selectedNode)) {
      this.selectedNode = newList[0].id; // Default to the first node if current selection is invalid
      this.updateGraph();
    }
  },
  selectedScenario(newScenario, oldScenario) {
    if (newScenario !== oldScenario) {
      this.updateMatrices();
    }
  }
}
}

</script>

<style>
.button-container, .reset-container {
  display: flex;
  flex-direction: column; /* Stack buttons vertically */
  align-items: flex-start; /* Align buttons to the left */
  justify-content: center; /* Center the container vertically */
  position: absolute;
  left: 5%; /* Same distance from the left */
  padding: 10px; /* Consistent padding */
  max-width: 200px; /* Uniform width */
  z-index: 1000;
  top:50%; /* Moves buttons around the z-axis */
}

.matrix-button, .run-button button {
  appearance: none;
  background-color: #FAFBFC;
  border: 1px solid rgba(27, 31, 35, 0.15);
  border-radius: 6px;
  box-shadow: rgba(27, 31, 35, 0.04) 0 1px 0, rgba(255, 255, 255, 0.25) 0 1px 0 inset;
  box-sizing: border-box;
  color: #24292E;
  cursor: pointer;
  display: inline-block;
  font-family: "apple-system", system-ui, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  font-size: 16px;
  font-weight: 500;
  line-height: 20px;
  list-style: none;
  margin-bottom: 10px;
  padding: 6px 16px;
  position: relative;
  transition: background-color 0.2s cubic-bezier(0.3, 0, 0.5, 1);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: middle;
  width: 110%;
  white-space: nowrap;
  word-wrap: break-word;
}

.matrix-button:hover, .run-button button:hover {
  background-color: #838282;
  color: white;
  text-decoration: none;
  transition-duration: 0.1s;
}

.matrix-button:active {
  background-color: #EDEFF2;
  box-shadow: rgba(225, 228, 232, 0.2) 0 1px 0 inset;
  transition: none 0s;
}

.matrix-button:focus {
  outline: 1px transparent;
}

.matrix-button:disabled:active {
  pointer-events: none;
}

.matrix-button:disabled:hover {
  box-shadow: none;
}

.active {
  background-color: #838282; /* Example: Green background for active button */
  color: white;
}

.dropdown-container {
  position: relative;
  left: 0%;
  top: 0%; /* Adjust as needed for proper vertical alignment */
  margin-bottom: 0px; /* Adds space between dropdowns */
  width: 100%; /* Ensures dropdowns do not overflow their container */
  display: inline-block /* Makes dropdown fill the line */
}

.dropdown-container label {
  /* Style for the label */
  background-color: #FAFBFC;
  border: 1px solid rgba(27, 31, 35, 0.15);
  border-radius: 6px;
  box-shadow: rgba(27, 31, 35, 0.04) 0 1px 0, rgba(255, 255, 255, 0.25) 0 1px 0 inset;
  box-sizing: border-box;
  color: #24292E;
  cursor: pointer;
  
  font-family: -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  font-size: 20px;
  font-weight: 500;
  line-height: 20px;
  list-style: none;
  margin-bottom: 10px;
  max-width: 250px;
  padding: 6px 16px;
  transition: background-color 0.2s cubic-bezier(0.3, 0, 0.5, 1);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: middle;
  width: 100%;
  white-space: wrap;
  word-wrap: break-word;
  position: absolute;
  left: 0%;
  top: 0%;
  z-index: 1000;
}

.dropdown-container select {
  display: flex;
  align-items: flex-start; /* Align buttons to the left */
  justify-content: center; /* Center the container vertically */
  position:absolute; /* Position the container absolutely */
  left: 0%; /* Distance from the left edge of the page */
  top: 0%; /* Start at the vertical center of the page */
  padding: 10px; /* Optional padding around the buttons */
  max-width: 200px; /* Maximum width of the button container */
  z-index: 1000;
}

.dropdown-container select {
  background-color: #FAFBFC;
  border: 1px solid rgba(27, 31, 35, 0.15);
  border-radius: 6px;
  box-shadow: rgba(27, 31, 35, 0.04) 0 1px 0, rgba(255, 255, 255, 0.25) 0 1px 0 inset;
  box-sizing: border-box;
  color: #24292E;
  cursor: pointer;
  display: block;
  font-family: -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  font-size: 20px;
  font-weight: 500;
  line-height: 20px;
  list-style: none;
  padding: 4px;
  position: relative;
  transition: background-color 0.2s cubic-bezier(0.3, 0, 0.5, 1);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  
  width: 100%;
  white-space: nowrap;
  word-wrap: break-word;
  margin: auto; /* Centers the select box if container width is adjusted */
}

.dropdown-container select:hover, .dropdown-container select:focus {
  border-color: #888;
}

/* Simulation Results Title */
h2 {
  font-family: -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  font-size: 24px;
  font-weight: bold;
  color: #333;
  margin-top: 20px; /* Adjust space above the title */
}

.editors {
  position: fixed;
  top: 20px;
  right: 20px;
  
  width: 310px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  z-index: 1500; /* Ensure it's on top of other elements */
  font-family: -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  color: #333; /* Set the text color */
}

.edit-modal {
  background-color: #FAFBFC; 
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 10px;  
}

.editors h3 {
  font-size: 18px; /* Set font size for headers */
  color: #24292e; /* Set a specific color for headers */
  font-weight: bold; /* Make headers bold */
}

.editors p {
  font-size: 16px; /* Set font size for paragraphs */
  color: #555; /* Set a different color for paragraphs */
}

.editors input, .editors button {
  margin-top: 2px;
  padding: 4px;
  width: 100%;
  box-sizing: border-box; /* Ensures padding does not affect width */
}

.editors button {
  background-color: #555555;
  color: white;
  border: none;
  cursor: pointer;
}

.editors button:hover {
  background-color: #96c1fa;
}

.add-node {
  background-color: #FAFBFC;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 10px;
}

.vis-network div.vis-label {
  text-align: center;
  line-height: 1.2;
  white-space: pre-wrap; /* Allows the text to wrap within the node */
}

.vis-network .vis-node {
  width: 80px; /* Ensure all nodes have the same width */
  height: 80px; /* Ensure all nodes have the same height */
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}

</style>