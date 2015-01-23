

window.TransferSet = function() {
	this.transfers = [];
}

TransferSet.prototype.disconnectTransfers = function() {
	for (var i = 0; i < this.transfers.length; i++) this.transfers[i].conn.disconnect();
}

TransferSet.prototype.reconnectTransfers = function() {
	for (var i = 0; i < this.transfers.length; i++) {
		var t = this.transfers[i], inLayer = getLayer(DISPLAY.anns[t.inAnnId], t.inLayerStage),
		outLayer = getLayer(DISPLAY.anns[t.outAnnId], t.outLayerStage);
		t.inNodeInLayer = t.inNodeInLayer % inLayer.length;
		t.outNodeInLayer = t.outNodeInLayer % outLayer.length;
		t.conn = getOrCreateConnection(inLayer[t.inNodeInLayer], outLayer[t.outNodeInLayer], t.getWeight());
	}
}

TransferSet.prototype.addTransfer = function(transfer) {
	if (!contains(transfer, this.transfers)) this.transfers.push(transfer);
}

TransferSet.prototype.toString = function() {
	var result = [];
	for (var i = 0; i < this.transfers.length; i++) {
		var t = this.transfers[i];
		result.push(t.inAnnId, "-", t.inLayerStage, "-", t.inNodeInLayer, "	");
	}
	result.push('\n');
	for (var i = 0; i < this.transfers.length; i++) {
		var t = this.transfers[i];
		result.push(t.outAnnId, "-", t.outLayerStage, "-", t.outNodeInLayer, "	");
	}
	result.push('\n');
	for (var i = 0; i < this.transfers.length; i++) result.push(Math.round(this.transfers[i].getWeight()*100)/100,"	")
	result.push('\n');
	return result.join('');
}

DEFAULT_TRANSFER_SET = new TransferSet();

window.Transfer = function(inAnnId, inLayerStage, inNodeInLayer, outAnnId, outLayerStage, outNodeInLayer, conn) {
	this.inAnnId = inAnnId;
	this.inLayerStage = inLayerStage;
	this.inNodeInLayer = inNodeInLayer;
	this.outAnnId = outAnnId;
	this.outLayerStage = outLayerStage;
	this.outNodeInLayer = outNodeInLayer;
	this.conn = conn;
}

Transfer.prototype.copy = function() {
	return new Transfer(this.inAnnId, this.inLayerStage, this.inNodeInLayer,
			this.outAnnId, this.outLayerStage, this.outNodeInLayer, this.conn);
}
Transfer.prototype.getWeight = function() {
	return isNan(this.weight) ? this.conn.weight : this.weight;
}
Transfer.prototype.multiplyWeight = function(multiplier) {
	this.weight = isNan(this.weight) ? this.conn.weight : this.weight;
	this.weight *= multiplier;
}

function connectNodes(ts, inANN, inStage, inNode, outANN, outStage, outNode, weightIter) {
	var ins = getLayer(inANN, inStage),
	outs = getLayer(outANN, outStage),
	inny = ins[inNode ? inNode % ins.length : randomInt(ins.length)],
	outy = outs[outNode ? outNode % outs.length : randomInt(outs.length)];
	var conn = getOrCreateConnection(inny, outy, weightIter ? weightIter.next() : null),
	transfer = new Transfer(inny.ann.focusId, inStage, inny.nodeInLayer, 
			outy.ann.focusId, outStage, outy.nodeInLayer, conn);
	ts.addTransfer(transfer);
}

function connectLayers(ts, inANN, inStage, outANN, outStage, weightIter) {
	var ins = getLayer(inANN, inStage),
	outs = getLayer(outANN, outStage);
	for (var i = 0; i < ins.length; i++) {
		var inny = ins[i];
		for (var j = 0; j < outs.length; j++) {
			var outy = outs[j];
			var conn = getOrCreateConnection(inny, outy, weightIter ? weightIter.next() : null),
			transfer = new Transfer(inny.ann.focusId, inStage, inny.nodeInLayer, 
					outy.ann.focusId, outStage, outy.nodeInLayer, conn);
			ts.addTransfer(transfer);
		}
	}
}
function connectANNs(borrower, lender, bInStage, bOutStage, lInStage, lOutStage, weightIter) {
	if (bInStage >= bOutStage || lInStage >= lOutStage) {console.log("CRITICAL ERROR: invalid stage"); return;}
	connectLayers(DEFAULT_TRANSFER_SET, borrower, bInStage, lender, lInStage);
	connectLayers(DEFAULT_TRANSFER_SET, lender, lOutStage, borrower, bOutStage);
}
function getLayer(ann, stage) {
	if (stage == 0) return ann.inputs;
	if (stage > ann.hiddenLayers.length) return ann.outputs;
	return ann.hiddenLayers[stage - 1];
}