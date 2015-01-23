

// MANAGEMENT

function createTrainingSampleDivs(data) {
	var result = [], splitData = data.split('\n');
	for (var i = 0; i < splitData.length; i++) {
		if (!splitData[i].length) continue;
		result.push(createTrainingSampleDiv(splitData[i]));
	}
	return result.join('');
}
function createTrainingSampleDiv(datum) {
	var ins = datum.split(':')[0].split(',');
	var outs = datum.split(':')[1].split(',');
	var result = ['<div ins="' + ins + '" outs="' + outs + '"><table><tr><td><table><tr>'];
	for (var i = 0; i < ins.length; i ++) result.push('<td>' + ins[i]);
	result.push('</table><td style="width:20%">:<td><table><tr>');
	for (var i = 0; i < outs.length; i ++) result.push('<td>' + outs[i]);
	result.push('</table><td><button onclick=ffT([', ins, '])>F</button>',
			'<td><button onclick=bbT([',ins,'],[',outs,'])>B</button>',
			'<td><button>X</button></table></div>');
	return result.join('');
}
function feedBlanks() {
	for (var i = 0; i < DISPLAY.anns.length; i++) {
		var blankIns = [], ann = DISPLAY.anns[i];
		for (var i = 0; i < ann.inputs.length; i++) blankIns.push(0);
		ann.feedForward(blankIns);
	}
}
function ffT(ins) {
	feedBlanks();
	ann1().feedForward(ins);
	DISPLAY.redraw();
}
function bbT(ins, outs) {
	feedBack(ann1(), outs);
	ffT(ins);
}

var DISPLAY = new ANNDisplay(el("displayDiv"));
DISPLAY.setANN();
function ann1() {return DISPLAY.anns[FOCUS]};
var genAlgo = new GeneticAlgo();
	
function getData() {
	var data = el('samples').children, result = [];
	for (var i = 0; i < data.length; i++) {
		var d = data[i], ins = d.getAttribute("ins").split(','), outs = d.getAttribute("outs").split(',');
		result.push({ins:ins,outs:outs});
	}
	return result;
}

train = function(data, fn, epochs, redrawInterval) {
	for (var epoch = 0; epoch < epochs; epoch++) fn(data);
}

el('addANNBtn').onclick = function() {USE_BIAS = el('useBias').checked; DISPLAY.addANN();};
el('createANNBtn').onclick = function() {USE_BIAS = el('useBias').checked; DISPLAY.setANN()};
el('createPOPBtn').onclick = function() {
	USE_BIAS = el('useBias').checked;
	var n = el('popSize').value;
	POPULATION = [];
	genAlgo.maxPop = n;
	if (genAlgo.isReuseAlgo && DISPLAY.anns.length < 2) {alert("ERROR: YOU NEED ANOTHER NETWORK FOR REUSE"); return;}
	for (var i = 0; i < n; i++) {
		var genome = genAlgo.createRandomGenome();
		addToPop(genome, genAlgo.fitnessFunction, getData());
	}
	genAlgo.displayPop();
};
el('createTrainingData').onclick = function() {
	el('samples').innerHTML += createTrainingSampleDivs(el('trainingDataArea').value);
	ann1().calcAccuracy();
}
el('clearTrainingData').onclick = function() {
	el('samples').innerHTML = "";
	console.log("cleartraining")
	ann1().calcAccuracy();
}
el('trainingDataArea').value ="0,0:0\r\n1,0:1\r\n0,1:1\r\n1,1:0\r\n";
el('trainANNBtn').onclick = function() {
	var ann = ann1();
	LEARNING_RATE = el('learnRate').value;
	train(getData(), function(data) {
		trainBackProp(ann, data)
	}, el('bpEpochs').value);
	DISPLAY.rescaleWeights();
	ann.calcAccuracy();
}
el('trainEVOBtn').onclick = function() {
	genAlgo.clearParams();
	train(getData(), function(data) {
		genAlgo.trainEvolution(data)
	}, el('gaEpochs').value);
	genAlgo.displayPop();
	DISPLAY.rescaleWeights();
	ann1().calcAccuracy();
}

el('connectANNBtn').onclick = function() {
	createTransfers();
}
function createTransfers() {
	var lenderId = el('lenderId').value,
	lenderIO = el('lenderIO').value.split(','),
	borrowerIO = el('borrowerIO').value.split(','),
	errorEl = el('connectError');
	if (lenderId < 0 || lenderId == FOCUS || lenderId >= DISPLAY.anns.length) errorEl.innerHTML = 'Error: invalid lender ID';
	else if (borrowerIO.length != 2 || lenderIO.length != 2) errorEl.innerHTML = 'Error: invalid I/O layers';
	else {
		errorEl.innerHTML = '';
		connectANNs(ann1(), DISPLAY.anns[lenderId], borrowerIO[0], borrowerIO[1], lenderIO[0], lenderIO[1]);
		DISPLAY.redraw();
	}
}

var evoSwitch1 = el('evoTypeSwitch1'), evoSwitch2 = el('evoTypeSwitch2');
evoSwitchFn = function() {
	if (evoSwitch1.innerHTML == "Net Evo") {
		evoSwitch1.innerHTML = "Reuse Evo";
		evoSwitch2.innerHTML = "Reuse Evo";
		el('mutWLabel').innerHTML = "weight mutation:";
		el('mutNLabel').innerHTML = "I/O mutation:";
		el('mutLLabel').innerHTML = "size mutation:";
		genAlgo = REUSE_GA;
	} else {
		evoSwitch1.innerHTML = "Net Evo";
		evoSwitch2.innerHTML = "Net Evo";
		el('mutWLabel').innerHTML = "weight mutation:";
		el('mutNLabel').innerHTML = "node mutation:";
		el('mutLLabel').innerHTML = "layer mutation:";
		genAlgo = new GeneticAlgo();
	}
}
evoSwitch1.onclick = evoSwitchFn;
evoSwitch2.onclick = evoSwitchFn;

var INPUT_MATRIX = [];
el('showInputMatrix').onclick = function() {
	INPUT_MATRIX = [];
	var html = ['<table style="border-spacing:initial">'], rows = el('imRows').value, cols = el('imCols').value;
	for (var r = 0; r < rows; r++) {
		html.push('<tr>');
		for (var c = 0; c < cols; c++) {
			var id = r * cols + c;
			INPUT_MATRIX[id] = 0;
			html.push('<td style="width:1em; height:1.2em; border: 1px solid grey" id=im', id, '>');
		}
	}
	html.push('</table><button id=addIM>Add as</button>output:<input id=imOut value="0,1" style="width:3em"></input>');
	el('inputMatrix').innerHTML = html.join('');
	for (var i = 0; i < INPUT_MATRIX.length; i++) {
		el('im' + i).onclick = function() {
			var thisEl = this, id = this.id.split('im')[1];
			if (INPUT_MATRIX[id] == 0) {
				INPUT_MATRIX[id] = 1;
				thisEl.style['background-color'] = 'rgba(0,0,0,1)';
			} else {
				INPUT_MATRIX[id] = 0;
				thisEl.style['background-color'] = 'rgba(255,255,255,1)';
			}
		}
	}
	el('addIM').onclick = function() {
		var out = el('imOut').value.split(','), vector = [INPUT_MATRIX[0]];
		for (var i = 1; i < INPUT_MATRIX.length; i++) vector.push(',', INPUT_MATRIX[i]);
		vector.push(':', out[0]);
		for (var i = 1; i < out.length; i++) vector.push(',', out[i]);
		vector.push('\r\n');
		el('trainingDataArea').value += vector.join('');
	}
}

//giveRandomOutputs(fillInputs(3));
window.addEventListener("resize", function() {DISPLAY.redraw();});

window.Iterator = function(array) {
	this.array = array;
	this.i = -1;
}
Iterator.prototype.next = function() {
	this.i++;
	return (this.i < this.array.length ? this.array[this.i] : null);
}

function debugStringAllConnections() {
	var result = [];
	for (var i = 0; i < DISPLAY.anns.length; i++) {
		result.push("A", i,":");
		var ics = DISPLAY.anns[i].allIncomingConnections();
		for (var j = 0; j < ics.length; j++) {
			var ic = ics[j];
			if (ic.inputNode.ann != ic.outputNode.ann) result.push("TTT");
			result.push("N", ic.inputNode.layerNumber, "/", ic.inputNode.nodeInLayer, "w", Math.floor(ic.weight*100));
		}
	}
	return result.join('');
}