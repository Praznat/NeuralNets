

window.TransferSet = function() {
	this.transfers = [];
}

DEFAULT_TRANSFER_SET = new TransferSet();

TransferSet.prototype.disconnectTransfers = function() {
	for (var i = 0; i < this.transfers.length; i++) this.transfers[i].conn.disconnect();
}

TransferSet.prototype.reconnectTransfers = function() {
	for (var i = 0; i < this.transfers.length; i++) {
		var t = this.transfers[i], inLayer = getLayer(DISPLAY.anns[t.inAnnId], t.inLayerStage),
		outLayer = getLayer(DISPLAY.anns[t.outAnnId], t.outLayerStage);
		if (inLayer.length <= t.inNodeInLayer || outLayer.length <= t.outNodeInLayer) continue;
		t.conn = getOrCreateConnection(inLayer[t.inNodeInLayer], outLayer[t.outNodeInLayer], t.conn.weight);
	}
}

window.Transfer = function(inAnnId, inLayerStage, inNodeInLayer, outAnnId, outLayerStage, outNodeInLayer, conn) {
	this.inAnnId = inAnnId;
	this.inLayerStage = inLayerStage;
	this.inNodeInLayer = inNodeInLayer;
	this.outAnnId = outAnnId;
	this.outLayerStage = outLayerStage;
	this.outNodeInLayer = outNodeInLayer;
	this.conn = conn;
}



function connectNodes(ts, inANN, inStage, outANN, outStage, weightIter) {
	var ins = getLayer(inANN, inStage),
	outs = getLayer(outANN, outStage);
	for (var i = 0; i < ins.length; i++) {
		var inny = ins[i];
		if (inny.isBias) continue;
		for (var j = 0; j < outs.length; j++) {
			var outy = outs[j];
			var conn = getOrCreateConnection(inny, outy, weightIter ? weightIter.next() : null),
			transfer = new Transfer(inny.ann.focusId, inStage, inny.nodeInLayer, 
					outy.ann.focusId, outStage, outy.nodeInLayer, conn);
			if (!contains(transfer, ts.transfers)) ts.transfers.push(transfer);
		}
	}
}
function connectANNs(borrower, lender, bInStage, bOutStage, lInStage, lOutStage, weightIter) {
	if (bInStage >= bOutStage || lInStage >= lOutStage) {console.log("CRITICAL ERROR: invalid stage"); return;}
	connectNodes(DEFAULT_TRANSFER_SET, borrower, bInStage, lender, lInStage);
	connectNodes(DEFAULT_TRANSFER_SET, lender, lOutStage, borrower, bOutStage);
}
function getLayer(ann, stage) {
	if (stage == 0) return ann.inputs;
	if (stage > ann.hiddenLayers.length) return ann.outputs;
	return ann.hiddenLayers[stage - 1];
}