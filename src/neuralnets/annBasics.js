var CANVAS;

console.log("ANN");

var GLOBAS_BIAS = 0.5;
var SIGMOID = function(x) {
	return 1 / (1 + Math.exp(GLOBAS_BIAS - x));
}
var RANDOM_WEIGHT = function() {
	return oneOf([-1, -.5, -.2, 0, .2, .5, 1]) / W_MULT;
}

var BLACK = 'rgba(0,0,0,1)';
var WHITE = 'rgba(255,255,255,1)';
var USE_BIAS = false;
var MAX_ABS_WEIGHT = 5;
var FOCUS = 0;
var W_MULT = 1;

function colorActivation(x) {
	return 'rgba(0,0,255,'+x+')';
}
function colorWeight(x) {
	var pre = x > 0 ? 'rgba(0,255,0,' : 'rgba(255,0,0,';
	return pre + Math.abs(x) + ')';
}
function lineWidthWeight(x) {
	return Math.round(3 * Math.abs(x)) + 1;
}

window.ANNDisplay = function(parentElement) {

	this.parent = parentElement;
	CANVAS = this.canvas = document.createElement('canvas');
	this.ctx = this.canvas.getContext('2d');
	var deviceRatio = window.devicePixelRatio || 1;
	var storeRatio = (this.ctx.webkitBackingStorePixelRatio ||
			this.ctx.mozBackingStorePixelRatio ||
			this.ctx.msBackingStorePixelRatio ||
			this.ctx.oBackingStorePixelRatio ||
			this.ctx.backingStorePixelRatio || 1);
	this.pixelMult = deviceRatio / storeRatio;

	parentElement.appendChild(this.canvas);
	this.anns = [];
	
	
	CANVAS.addEventListener('click', function(event) {
		FOCUS = Math.floor(event.offsetY / (CANVAS.offsetHeight / DISPLAY.anns.length));
		DISPLAY.redraw();
		DISPLAY.anns[FOCUS].calcAccuracy();
		highlightNode(event.offsetX, event.offsetY);
	}, false);
	
}

function highlightNode(mx, my) {
	
	var ann = DISPLAY.anns[FOCUS], allLayers = [ann.inputs];
	allLayers = allLayers.concat(ann.hiddenLayers).concat([ann.outputs]);
	for (var i = 0; i < allLayers.length; i++) {
		var layer = allLayers[i];
		for (var j = 0; j < layer.length; j++) {
			var node = layer[j];
			if (node.x <= mx && node.x + node.radius * 2 >= mx
					&& node.y - node.radius <= my && node.y + node.radius >= my) {
				el('hlNode').innerHTML = "L"+node.layerNumber + "N"+node.nodeInLayer
					+ " a:" + Math.round(node.activation*100)/100;
				console.log(node);
				return;
			}
		}
	}
}

ANNDisplay.prototype.setANN = function(ann) {
	var a = ann || createANN();
	DEFAULT_TRANSFER_SET.disconnectTransfers();
	this.anns[FOCUS] = a;
	a.focusId = FOCUS;
	DEFAULT_TRANSFER_SET.reconnectTransfers();
	this.redraw();
	a.calcAccuracy();
	return a;
}
ANNDisplay.prototype.addANN = function() {
	var newH = this.parent.offsetHeight * (1 + ++FOCUS) / this.anns.length + "px";
	this.parent.style.height = newH;
	this.canvas.style.height = newH;
	this.setANN();
}

function createANN() {
	var numIn = el('numInputs').value;
	var numHidden = el('numHiddens').value.split(',');
	var numOut = el('numOutputs').value;
	var result = createFFNeuralNetwork(numIn, numHidden, numOut);
	return result;
}

ANNDisplay.prototype.redraw = function() {
	var q = this.canvas, c = this.ctx;
	var w = q.width = this.parent.offsetWidth * this.pixelMult;
	var h = q.height = this.parent.offsetHeight * this.pixelMult;
	q.style.width = this.parent.offsetWidth + 'px';
	q.style.height = this.parent.offsetHeight + 'px';
	var hh = h / this.anns.length;
	
	for (var i = 0; i < this.anns.length; i++) this.anns[i].draw(c, 0, i * hh, w, hh);
}

ANNDisplay.prototype.rescaleWeights = function() {
	var maxAbsWgt = 0, conns = [];
	for (var i = 0; i < this.anns.length; i++) {
		conns = conns.concat(this.anns[i].allIncomingConnections());
	}
	for (var i = 0; i < conns.length; i++) {
		var conn = absWgt = Math.abs(conns[i].weight);
		if (absWgt > maxAbsWgt) maxAbsWgt = absWgt;
	}
//	if (maxAbsWgt < MAX_ABS_WEIGHT) return this;
	var mult = MAX_ABS_WEIGHT / maxAbsWgt;
	W_MULT = mult;
//	for (var i = 0; i < conns.length; i++) conns[i].weight *= mult;
//	GLOBAS_BIAS *= mult;
}

window.FFNeuralNetwork = function() {}
function createFFNeuralNetwork(numInputs, numHiddenNodesArray, numOutputs) {
	var ann = new FFNeuralNetwork();
	ann.inputs = [];
	if (USE_BIAS) ann.bias = createInputNode(ann, -1, -1, SIGMOID, true);
	for (var i = 0; i < numInputs; i++) ann.inputs.push(createInputNode(ann, 0, i, SIGMOID));
	ann.hiddenLayers = [];
	var prevNodes = ann.inputs;
	for (var i = 0; i < numHiddenNodesArray.length; i++) {
		var hiddenNodes = [];
		for (var j = 0; j < numHiddenNodesArray[i]; j++) hiddenNodes.push(
				new HiddenNode(ann, i+1, j, SIGMOID, ann.bias));
		ann.hiddenLayers.push(hiddenNodes);
		for (var k = 0; k < hiddenNodes.length; k++) {
			for (var kk = 0; kk < prevNodes.length; kk++) {
				var pn = prevNodes[kk];
				if (!pn.isBias) getOrCreateConnection(pn, hiddenNodes[k]);
			}
		}
		if (hiddenNodes.length) prevNodes = hiddenNodes;
	}
	ann.outputs = [];
	for (var i = 0; i < numOutputs; i++) {
		ann.outputs.push(new HiddenNode(ann, ann.hiddenLayers.length + 1, i, SIGMOID, ann.bias));
		for (var j = 0; j < prevNodes.length; j++) {
			getOrCreateConnection(prevNodes[j], ann.outputs[i]);
		}
	}
	return ann;
}
FFNeuralNetwork.prototype.redraw = function() {this.draw(this.lastCtx, this.lastX, this.lastY, this.lastW, this.lastH);}
FFNeuralNetwork.prototype.draw = function(ctx, x, y, w, h) {
	this.lastCtx = ctx; this.lastX = x; this.lastY = y; this.lastW = w; this.lastH = h;
	ctx.font = "18px Arial";

	ctx.fillStyle = 'rgba(240,240,240,1)';
//	ctx.fillRect(x,y,w,h);
	if (FOCUS == this.focusId) {
		ctx.strokeStyle = 'rgba(0,0,240,1)';
		ctx.strokeRect(x,y,w,h);
	}

	var nonBiasN = 0;
	for (var i = 0; i < this.inputs.length; i++) if (!this.inputs[i].isBias) nonBiasN++;
	var numNodeColumns = 1 + this.hiddenLayers.length + 1;
	var numNodeRows = Math.max(nonBiasN, this.outputs.length);
	for (var i = 0; i < this.hiddenLayers.length; i++) numNodeRows = Math.max(numNodeRows, this.hiddenLayers[i].length);

	var cw = Math.max(6, w / (1 + 2 * numNodeColumns)), rh = Math.max(6, h / (1 + 2 * numNodeRows)), radius = Math.min(cw, rh) / 2;
	
	var toDraw = [];
	//outputs
	var rho = h / (2 * this.outputs.length);
	for (var i = 0; i < this.outputs.length; i++) {
		toDraw.push(this.outputs[i].predraw(w - cw, y + rho * (1 + 2 * i), radius));
	}
	//hiddens
	for (var i = 0; i < this.hiddenLayers.length; i++) {
		var hiddenNodes = this.hiddenLayers[i];
		var rhh = h / (2 * hiddenNodes.length);
		for (var j = 0; j < hiddenNodes.length; j++) {
			toDraw.push(hiddenNodes[j].predraw(cw * (3 + 2 * i), y + rhh * (1 + 2 * j), radius));
		}
	}
	//inputs
	var rhi = h / (2 * nonBiasN), b = 0;
	for (var i = 0; i < this.inputs.length; i++) {
		var inny = this.inputs[i];
		toDraw.push(inny.predraw(cw, y + rhi * (1 + 2 * (i - b)), rhi / 2, rhi / 2));
	}
	//connections TODO consolidate using concat
	var connectionsToDraw = [];
	for (var i = 0; i < this.inputs.length; i++) {
		var ip = this.inputs[i];
		if (!ip.inputConnections) break;
		for (var j = 0; j < ip.inputConnections.length; j++) connectionsToDraw.push(ip.inputConnections[j]);
	}
	for (var i = 0; i < this.hiddenLayers.length; i++) {
		var hiddenNodes = this.hiddenLayers[i];
		for (var j = 0; j < hiddenNodes.length; j++) {
			var hn = hiddenNodes[j];
			for (var k = 0; k < hn.inputConnections.length; k++) connectionsToDraw.push(hn.inputConnections[k]);
		}
	}
	for (var i = 0; i < this.outputs.length; i++) {
		var op = this.outputs[i];
		for (var j = 0; j < op.inputConnections.length; j++) connectionsToDraw.push(op.inputConnections[j]);
	}
	connectionsToDraw.sort(CONNECTION_COMPARATOR);
	for (var i = 0; i < connectionsToDraw.length; i++) connectionsToDraw[i].draw(ctx)
	
	for (var i = 0; i < toDraw.length; i++) toDraw[i].draw(ctx);

}
FFNeuralNetwork.prototype.feedForward = function(ins) {
	var n = Math.min(ins.length, this.inputs.length), nextNodes = [], nowNodes = [];
	var seenNodes = []; // to prevent infinite loop from recurrency
	
	for (var i = 0; i < n; i++) {
		var inny = this.inputs[i], outConns = inny.outputConnections;
		if (contains(inny, seenNodes)) continue;
		else seenNodes.push(inny);
		inny.activate(parseInt(ins[i]));
		for (j = 0; j < outConns.length; j++) {
			var outy = outConns[j].outputNode;
			if (!contains(outy, nextNodes)) nextNodes.push(outy);
		}
	}
	while (nextNodes.length) {
		nowNodes = [];
		for (var i = 0; i < nextNodes.length; i++) nowNodes[i] = nextNodes[i];
		nextNodes = [];
		for (var i = 0; i < nowNodes.length; i++) {
			var inny = nowNodes[i], outConns = inny.outputConnections;
			if (contains(inny, seenNodes)) continue;
			else seenNodes.push(inny);
			inny.activate();
			for (j = 0; j < outConns.length; j++) {
				var outy = outConns[j].outputNode;
				if (!contains(outy, nextNodes)) nextNodes.push(outy);
			}
		}
	}
}
//FFNeuralNetwork.prototype.feedForwardOLD = function(ins) {
//	var n = Math.min(ins.length, this.inputs.length);
//	for (var i = 0; i < n; i++) this.inputs[i].activate(parseInt(ins[i]));
//	for (var i = 0; i < this.hiddenLayers.length; i++) {
//		var hiddenNodes = this.hiddenLayers[i];
//		for (var j = 0; j < hiddenNodes.length; j++) hiddenNodes[j].activate();
//	}
//	for (var i = 0; i < this.outputs.length; i++) this.outputs[i].activate();
//}

FFNeuralNetwork.prototype.oneDatumAvgSqrErr = function(ins, outs) {
	var sumSqrErr = 0;
	this.feedForward(ins);
	for (var i = 0; i < outs.length; i++) {
		var output = this.outputs[i];
		var err = (output ? output.activation : 0) - outs[i];
		sumSqrErr += err * err;
	}
	return sumSqrErr / outs.length;
}
FFNeuralNetwork.prototype.isCorrectClassification = function(ins, outs) {
	var n = Math.min(outs.length, this.outputs.length);
	this.feedForward(ins);
	var nnOuts = [];
	for (var i = 0; i < n; i++) nnOuts.push({value:this.outputs[i].activation, plc:i});
	nnOuts.sort(function(a, b) {return a.value < b.value ? 1 : -1;});
	for (var i = 0; i < n; i++) if (outs[nnOuts[i].plc] != Math.round(nnOuts[i].value)) return false;
	return true;
}

FFNeuralNetwork.prototype.calcMSE = function(data) {
	var sumSqrErr = 0;
	for (var i = 0; i < data.length; i++) {
		var d = data[i];
		sumSqrErr += this.oneDatumAvgSqrErr(d.ins, d.outs);
	}
	return Math.sqrt(sumSqrErr / data.length);
}
FFNeuralNetwork.prototype.calcHitRate = function(data) {
	var numCorrect = 0;
	for (var i = 0; i < data.length; i++) {
		var d = data[i];
		if (this.isCorrectClassification(d.ins, d.outs)) numCorrect++;
	}
	return numCorrect / data.length;
}
FFNeuralNetwork.prototype.calcAccuracy = function() {
	feedBlanks();
	var data = getData();
	el('focus').innerHTML = FOCUS;
	el('cError').innerHTML = Math.round(10000 * this.calcMSE(data)) / 10000;
	el('cRate').innerHTML = Math.round(10000 * this.calcHitRate(data)) / 100 + "%";
	DISPLAY.redraw();
}
FFNeuralNetwork.prototype.allIncomingConnections = function() {
	var nodeLayers = this.hiddenLayers.concat([this.outputs]), result = [];
	if (this.inputs.inputConnections) nodeLayers = nodeLayers.concat(this.inputs.inputConnections);
	for (var i = 0; i < nodeLayers.length; i++) {
		var layer = nodeLayers[i];
		for (var j = 0; j < layer.length; j++) result = result.concat(layer[j].inputConnections);
	}
	return result;
}
FFNeuralNetwork.prototype.numLayers = function(includeInputs) {
	return this.hiddenLayers.length + (includeInputs ? 2 : 1);
}


////////////////// CONNECTIONS
var ccc = 0;
window.Connection = function(weight) {
	this.weight = isNan(weight) ? RANDOM_WEIGHT() : weight;
	this.blameFromOutput = 0;
	this.input;
	this.output;
	this.id = ccc++;
}
Connection.prototype.connect = function(input, output) {
	if (this.input || this.output) console.log("CRITICAL ERROR: cannot reset connection I/O");
	this.inputNode = input;
	this.outputNode = output;
	this.inputNode.addOutputConnection(this);
	this.outputNode.addInputConnection(this);
}
Connection.prototype.disconnect = function() {
	 var ioc = indexOf(this, this.inputNode.outputConnections),
	 	oic = indexOf(this, this.outputNode.inputConnections);
	 if (ioc >= 0) this.inputNode.outputConnections.splice(ioc, 1);
	 if (oic >= 0) this.outputNode.inputConnections.splice(oic, 1);
}
Connection.prototype.draw = function(ctx) {
	drawLine(ctx, colorWeight(this.weight), lineWidthWeight(W_MULT * this.weight),
			this.inputNode.x + 2 * this.inputNode.radius, this.inputNode.y,
			this.outputNode.x, this.outputNode.y);
}
var CONNECTION_COMPARATOR = function(a, b) {
	return Math.abs(a.weight) < Math.abs(b.weight) ? -1 : 1;
}
function getOrCreateConnection(inputNode, outputNode, weight) { // only gives weight if this is new connection
	var conn = null;
	for (var i = 0; i < inputNode.outputConnections.length; i++) {
		var outputConn = inputNode.outputConnections[i];
		if (outputConn.outputNode == outputNode) {
			conn = outputConn;
			break;
		}
	}
	if (conn != null) { // just check input-output = output-input
		var isOK = false;
		for (var i = 0; i < outputNode.inputConnections.length; i++) {
			var inputConn = outputNode.inputConnections[i];
			if (inputConn.inputNode == inputNode && inputConn == conn) isOK = true;
		}
		if (!isOK) console.log("CRITICAL ERROR: input-output != output-input");
	}
	if (conn == null) {
		conn = new Connection(weight);
		conn.connect(inputNode, outputNode);
	}
	return conn;
}

//////////////////HIDDENS
var NID = 0;
window.HiddenNode = function(ann, layerNumber, nodeInLayer, activationFunction, bias) {
	this.id = NID++;
	this.ann = ann;
	this.layerNumber = layerNumber;
	this.nodeInLayer = nodeInLayer;
	this.inputConnections = [];
	this.outputConnections = [];
	this.activationFunction = activationFunction;
	this.activation = 0;
	if (bias) this.giveBias(bias);
}
HiddenNode.prototype.giveBias = function(bias, weight) {
	var conn = getOrCreateConnection(bias, this, weight);
}
HiddenNode.prototype.predraw = function(x, y, r) {this.x = x; this.y = y; this.radius = r; return this;}

HiddenNode.prototype.activate = function(x) {
	if (x) this.activation = x;
	else this.activation = 0;
	if (!this.inputConnections.length) return;
	var sum = 0;
	for (var i = 0; i < this.inputConnections.length; i++) {
		var ic = this.inputConnections[i];
		sum += ic.weight * ic.inputNode.activation;
	}
	this.activation = this.activationFunction(sum);
}
HiddenNode.prototype.addInputConnection = function(connection) {
	this.inputConnections.push(connection);
}
HiddenNode.prototype.addOutputConnection = function(connection) {
	this.outputConnections.push(connection);
}
HiddenNode.prototype.draw = function(ctx) {
	if (!this.inputConnections.length) { // is input
		drawTriangle(ctx, colorActivation(this.activation), this.x + this.radius*2, this.y, this.radius, this.radius);
	} else {
		drawCircle(ctx, colorActivation(this.activation), this.x, this.y, this.radius);
		var txt = Math.round(this.activation * 100) / 100;
		ctx.fillStyle = WHITE;
		ctx.fillText(txt, this.x, this.y);
		ctx.strokeStyle = BLACK;
		ctx.strokeText(txt, this.x, this.y);
	}
}

createInputNode = function(ann, layerNumber, nodeInLayer, activationFunction, isBias) {
	node = new HiddenNode(ann, layerNumber, nodeInLayer, activationFunction);
	if (isBias) {
		node.isBias = true;
		node.activation = 1;
	}
	return node;
}

////////////////// INPUTS

window.InputNode = function(isBias) {
	this.outputConnections = [];
	if (isBias) this.isBias = true;
	this.activation = isBias ? 1 : 0;
}
InputNode.prototype.predraw = function(x, y, r) {this.x = x; this.y = y; this.radius = r; return this;}

InputNode.prototype.activate = function(x) {this.activation = x;}
InputNode.prototype.addOutputConnection = function(connection) {
	this.outputConnections.push(connection);
}
InputNode.prototype.draw = function(ctx) {
	drawTriangle(ctx, colorActivation(this.activation), this.x, this.y, this.radius, this.radius);
}

function drawLine(ctx, color, wgt, x0, y0, x1, y1) {
	ctx.strokeStyle = color;
	ctx.lineWidth = wgt;
	ctx.beginPath();
	ctx.moveTo(x0, y0);
	ctx.lineTo(x1, y1);
	ctx.stroke();
	ctx.closePath();
	ctx.lineWidth = 1;
}

function drawCircle(ctx, color, x, y, radius) {
	fillCircle(ctx, WHITE, x, y, radius);
	fillCircle(ctx, color, x, y, radius);
	strokeCircle(ctx, x, y, radius);
}
function fillCircle(ctx, color, x, y, radius) {
	ctx.fillStyle = color;
	ctx.beginPath();
	ctx.arc(x + radius, y, radius, 0, 2*Math.PI, true);
	ctx.fill();
	ctx.closePath();
}
function strokeCircle(ctx, x, y, radius) {
	ctx.strokeStyle = BLACK;
	ctx.beginPath();
	ctx.arc(x + radius, y, radius, 0, 2*Math.PI, true);
	ctx.stroke();
	ctx.closePath();
}

function drawTriangle(ctx, color, x, y, w, h) {
	fillTriangle(ctx, WHITE, x, y, w, h);
	fillTriangle(ctx, color, x, y, w, h);
	strokeTriangle(ctx, x, y, w, h);
}
function fillTriangle(ctx, color, x, y, w, h) {
	if (!x || !y) return;
	ctx.fillStyle = color;
	ctx.beginPath();
	ctx.moveTo(x - w, y - h/2);
	ctx.lineTo(x - w, y + h/2);
	ctx.lineTo(x, y);
	ctx.fill();
	ctx.closePath();
}
function strokeTriangle(ctx, x, y, w, h) {
	if (!x || !y) return;
	ctx.strokeStyle = BLACK;
	ctx.beginPath();
	ctx.moveTo(x - w, y - h/2);
	ctx.lineTo(x - w, y + h/2);
	ctx.lineTo(x, y);
	ctx.lineTo(x - w, y - h/2);
	ctx.stroke();
	ctx.closePath();
}



