
var LEARNING_RATE = 1;


var TMPSMACK = [0,0];

function propagateErrorFromOutput(connection, delta) {
	if (connection.blameFromOutput) console.log("CRITICAL ERROR: should not reassign blame")
	var frozen = isFrozen(connection);
	connection.blameFromOutput = connection.weight * delta;
	if (!frozen) connection.weight += LEARNING_RATE * delta * connection.inputNode.activation;
	if (connection.inputNode.activation == 0) TMPSMACK[0] += delta;
	if (connection.inputNode.activation == 1) TMPSMACK[1] += delta;
}

function isFrozen(connection) {
	var freeANN = DISPLAY.anns[FOCUS],
	inANN = connection.inputNode.ann, outANN = connection.outputNode.ann;
	return inANN == outANN && inANN != freeANN; // free if cross-ANN or focus ANN
}

function trainBackProp(ann, data) {
	for (var i = 0; i < data.length; i++) {
		var d = data[i];
		ann.feedForward(d.ins);
		feedBack(ann, d.outs);
	}
//	console.log(TMPSMACK);
	TMPSMACK = [0,0];
}

function feedBack(ann, outTargets) {
	var n = Math.min(outTargets.length, ann.outputs.length),
		nowNodes = [], nextNodes = [], connsToClear = [];
	var seenNodes = []; // to prevent infinite loop from recurrency
	for (var i = 0; i < n; i++) {
		var node = ann.outputs[i], derivative = node.activation * (1 - node.activation), // TODO depend on actFn
			delta = derivative * (outTargets[i] - node.activation), inConns = node.inputConnections;
//		console.log(delta +","+ derivative +","+ outTargets[i] +","+ node.activation);
		if (contains(node, seenNodes)) continue;
		else seenNodes.push(node);
		for (var j = 0; j < inConns.length; j++) {
			var ic = inConns[j];
			propagateErrorFromOutput(ic, delta);
			connsToClear.push(ic);
			var inNode = ic.inputNode;
			if (inNode.ann.focusId == 0  && !contains(inNode, nextNodes)) nextNodes.push(inNode);
		}
	}
	while (nextNodes.length) {
		nowNodes = [];
//		console.log("---");
		for (var j = 0; j < nextNodes.length; j++) nowNodes[j] = nextNodes[j];
		nextNodes = [];
		for (var j = 0; j < nowNodes.length; j++) {
			var node = nowNodes[j], sumBlame = 0, outConns = node.outputConnections;
			if (contains(node, seenNodes)) continue;
			else seenNodes.push(node);
			for (var k = 0; k < outConns.length; k++) {
				sumBlame += outConns[k].blameFromOutput;
			}
			var derivative = node.activation * (1 - node.activation), // TODO depend on actFn
				delta = derivative * sumBlame, inConns = node.inputConnections;
//			console.log(node);
//			console.log(delta +","+ sumBlame +","+ node.activation);
			for (var k = 0; k < inConns.length; k++) {
				var ic = inConns[k];
				propagateErrorFromOutput(ic, delta);
				connsToClear.push(ic);
				var inNode = ic.inputNode;
				if (!contains(inNode, nextNodes)) nextNodes.push(inNode);
			}
		}
	}
	for (var i = 0; i < connsToClear.length; i++) connsToClear[i].blameFromOutput = 0;
}

