POPULATION = [];

FITNESS_COMPARATOR = function(a, b) { return a.fitness < b.fitness ? -1 : 1; };

window.GeneticAlgo = function() {
	this.fitnessFunction = function(ann, data) {
		return ann.calcMSE(data)
	}
}
GeneticAlgo.prototype.clearPop = function() {
	POPULATION = [];
}
GeneticAlgo.prototype.displayPop = function() {
	var box = el('populationBox'), html = [];
	for (var i = 0; i < POPULATION.length; i++) {
		var p = POPULATION[i], col = '';
		if (!p) continue;
		html.push('<td style="border: 1px solid black; cursor: pointer;background-color:', col, '" ',
			'onclick=highlight(getByIndex(', i, '))>',
			Math.round(100 * p.fitness) / 100);
	}
	box.innerHTML = html.join('');
	if (!POPULATION.length) {console.log("ERROR: YOU NEED A POPULATION FIRST"); return;}
	DISPLAY.setANN(POPULATION[0].genome);
}
function addToPop(genome, fitnessFn, data) {
	POPULATION.push({genome:genome, fitness:fitnessFn(genome, data)});
}
GeneticAlgo.prototype.clearParams = function() {
	this.killR = null;
	this.mutation = null;
}

function evolve(data, killRate, fitnessFn, newChildFn, newRandoFn) {

	var numSurvivors = Math.round(POPULATION.length * (1 - killRate));
	var numToRepopulate = POPULATION.length - numSurvivors;
	POPULATION.sort(FITNESS_COMPARATOR);
	POPULATION = POPULATION.slice(0, numSurvivors);
	for (var i = 0; i < numSurvivors; i++) { //replace survivors with fitter mutations
		var parent = POPULATION[i],
			childGenome = newChildFn(parent),
			child = {genome:childGenome, fitness:fitnessFn(childGenome, data)};
		if (FITNESS_COMPARATOR(child, parent) < 0) POPULATION[i] = child;
	}
	for (var i = 0; i < numToRepopulate; i++) { // reintroduce randos 
		addToPop(newRandoFn(), fitnessFn, data);
	}
	
}

GeneticAlgo.prototype.trainEvolution = function(data) {
	DEFAULT_TRANSFER_SET.disconnectTransfers();
	this.killR = this.killR ? this.killR
			: Math.min(1, Math.max(0, el('killRate').value / 100));
	this.mutation = this.mutation ? this.mutation
			: { wgt: Math.min(1, Math.max(0, el('mutationRateW').value / 100)),
				node: Math.min(1, Math.max(0, el('mutationRateN').value / 100)),
				layer: Math.min(1, Math.max(0, el('mutationRateL').value / 100)) }
	
	var ga = this;
	var newChildFn = function(parent) {
		var replication = ga.copyGenotype(parent.genome, ga.mutation.wgt, ga.mutation.node, ga.mutation.layer);
		return ga.fromGenotype(replication.offspring);
	};
	var newRandoFn = function() {
		return createANN();
	}
	
	evolve(data, this.killR, this.fitnessFunction, newChildFn, newRandoFn);
	
	DEFAULT_TRANSFER_SET.reconnectTransfers();
}

// gene encoding
var NEW_LAYER = 'L', NEW_NODE = 'n', NEW_WEIGHT = 'w', IS_BIAS = 'b', 
	MAX_NODES_PER_LAYER = 5, MAX_MIDDLE_LAYERS = 4,
	ROUNDY = function(x) {return Math.round(x * 10000) / 10000;};

function dnaLength(numInputs, numOutputs) {
	return MAX_NODES_PER_LAYER * (numInputs + numOutputs + MAX_NODES_PER_LAYER * (MAX_MIDDLE_LAYERS - 1));
}

function rMut(m) {return 2 * Math.random() * m - m;}
function dMut(m) {var r = Math.random(); return r < m ? (r < m/2 ? -1 : 1) : 0;}
function pushFor(v, x) {
	for (var i = 0; i < v.length; i++) v[i].push(x);
}

GeneticAlgo.prototype.copyGenotype = function(ann, mutateW, mutateN, mutateL) {
	var numInputs = ann.inputs.length, numOutputs = ann.outputs.length;
	var layers = ann.hiddenLayers.concat([ann.outputs]);
	var dmL = dMut(mutateL), isUsingBias = false, lastNodeDelta = 0;
	if ((dmL > 0 && ann.hiddenLayers.length >= MAX_MIDDLE_LAYERS) || (dmL < 0 && ann.hiddenLayers.length <= 1)) dmL = 0;
	
	var original = [], mutant = [];
	for (var i = 0; i < layers.length; i++) {
		var dnas = [original, mutant];
		if (i == layers.length - 1 && dmL < 0) dnas = [original]; // mutate lose layer
		pushFor(dnas, NEW_LAYER);
		var nodes = layers[i], dmN = dMut(mutateN), maxInputsSeen = 0;
		if ((dmN > 0 && nodes.length >= MAX_NODES_PER_LAYER) || (dmN < 0 && nodes.length <= 1)) dmN = 0;
		for (var j = 0; j < nodes.length; j++) {
			if (dmN < 0 && j == nodes.length - 1) dnas = [original]; // mutate lose node
			pushFor(dnas, NEW_NODE);
			var node = nodes[j], conns = node.inputConnections,
			connsToDelete = Math.max(-lastNodeDelta, 0);
			maxInputsSeen = Math.max(maxInputsSeen, conns.length + lastNodeDelta);
			for (var k = 0; k < conns.length; k++) {
				var conn = conns[k], w0 = ROUNDY(conn.weight), isBias = conn.inputNode.isBias,
					wm = (mutateW ? ROUNDY(conn.weight * Math.exp(rMut(mutateW))) : w0);
				if (isBias) isUsingBias = true;
				original.push(NEW_WEIGHT, (isBias ? IS_BIAS : ''), w0);
				// skip mutant weights if this layer or previous node was lost in mutation
				if (dnas.length < 2 || (!isBias && connsToDelete > 0)) {connsToDelete--; continue;}
				mutant.push(NEW_WEIGHT, (isBias ? IS_BIAS : ''), wm);
			}
			if (dnas.length > 1) for (var k = 0; k < lastNodeDelta; k++) {
				mutant.push(NEW_WEIGHT, RANDOM_WEIGHT());
			}
		}
		if (dmN > 0) { // mutate create node
			mutant.push(NEW_NODE);
			if (isUsingBias) {mutant.push(NEW_WEIGHT, IS_BIAS, RANDOM_WEIGHT());	maxInputsSeen--;}
			for (var k = 0; k < maxInputsSeen; k++) mutant.push(NEW_WEIGHT, RANDOM_WEIGHT());
		}
		lastNodeDelta = dmN;
	}
	if (dmL > 0) { // mutate create layer
		// create new layer same length as output layer AFTER old output layer (becomes new output layer)
		mutant.push(NEW_LAYER);
		for (var i = 0 ; i < ann.outputs.length; i++) {
			mutant.push(NEW_NODE);
			if (isUsingBias) {mutant.push(NEW_WEIGHT, IS_BIAS, RANDOM_WEIGHT());}
			for (var k = 0; k < ann.outputs.length + lastNodeDelta; k++) mutant.push(NEW_WEIGHT, RANDOM_WEIGHT());
		}
	}
//	console.log(original.join(''));
//	console.log(mutant.join(''));
//	console.log("   ");
	return {original:original.join(''), offspring:mutant.join('')};
}
GeneticAlgo.prototype.fromGenotype = function(dna) {
	var ann = new FFNeuralNetwork(), dnaLayers = dna.split(NEW_LAYER), doBiases = [];
	ann.inputs = []; ann.hiddenLayers = []; ann.outputs = [];
	dnaLayers = dnaLayers.splice(1, dnaLayers.length);
	for (var i = 0; i < dnaLayers.length; i++) {
		var dnaLayer = dnaLayers[i], dnaNodes = dnaLayer.split(NEW_NODE), nil = 0,
			onFirstHiddenLayer = i == 0, onOutputLayer = i == dnaLayers.length - 1;
		dnaNodes = dnaNodes.splice(1, dnaNodes.length);
		for (var j = 0; j < dnaNodes.length; j++) {
			var dnaNode = dnaNodes[j], dnaWeights = dnaNode.split(NEW_WEIGHT), annNode = new HiddenNode(ann, i+1, nil++, SIGMOID),
				annCurrentLayer = onOutputLayer ? ann.outputs : ann.hiddenLayers[i];
			if (annCurrentLayer) annCurrentLayer.push(annNode)
			else {annCurrentLayer = [annNode]; ann.hiddenLayers.push(annCurrentLayer);} // new hidden layer
			dnaWeights = dnaWeights.splice(1, dnaWeights.length);
			var rn = 0;
			for (var k = 0; k < dnaWeights.length; k++) {
				var w = dnaWeights[k];
				if (w.charAt(0) == IS_BIAS) doBiases.push({node:annNode,wgt:parseFloat(w.slice(1, w.length))});
				else { // not bias
					var annLeftLayer = getLeftLayer(ann, i), annLeftNode = annLeftLayer[rn];
					if (!annLeftNode) annLeftLayer[rn] = annLeftNode = (onFirstHiddenLayer ? createInputNode(ann, 0, nil++, SIGMOID)
						: new HiddenNode(ann, i, nil++, SIGMOID));
					getOrCreateConnection(annLeftNode, annNode, parseFloat(w));
					rn++;
				}
			}
		}
		annLeftLayer = ann.hiddenLayers[i]; 
	}
	for (var i = 0; i < doBiases.length; i++) {
		var bias = doBiases[i];
		bias.node.giveBias(ann.inputs, bias.wgt);
	}
	DISPLAY.rescaleWeights();
	return ann;
}
function getLeftLayer(ann, i) {
	var last = i == 0 ? ann.inputs : ann.hiddenLayers[i-1];
	return !last ? ann.inputs : last;
}
GeneticAlgo.prototype.mutateDNA = function(dna, mutateW, mutateN, mutateL) {
	var layers = dna.split(NEW_LAYER);
	for (var i = 0; i < layers.length; i++) {
		var layer = layers[i], nodes = layer.split(NEW_NODE);
	}
}

function highlight(member) {
	return DISPLAY.setANN(member.genome);
}
function getByIndex(i) {
	return POPULATION[i];
}