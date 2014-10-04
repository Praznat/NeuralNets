

REUSE_GA = new GeneticAlgo();

REUSE_GA.isReuseAlgo = true;

REUSE_GA.fitnessFunction = function(genome, data) {
	REUSE_GA.refreshTransfers(genome);
	feedBlanks();
	var result = ann1().calcMSE(data);
//	genome.debugString = debugStringAllConnections();
//	console.log(result);
//	console.log(genome.toString());
	return result;
};

REUSE_GA.refreshTransfers = function(newTransferSet) {
	if (!REUSE_GA.currDisplayTS) {
		REUSE_GA.currDisplayTS = newTransferSet;
	} else if (REUSE_GA.currDisplayTS != newTransferSet) {
		REUSE_GA.currDisplayTS.disconnectTransfers();
		REUSE_GA.currDisplayTS = newTransferSet;
	} else {return;}
	REUSE_GA.currDisplayTS.reconnectTransfers();
}

REUSE_GA.displayGenome = function(i) {
	REUSE_GA.refreshTransfers(POPULATION[i || 0].genome);
//	console.log(REUSE_GA.currDisplayTS.toString());
	ann1().calcAccuracy();
//	var debug = debugStringAllConnections();
//	console.log(debug);
//	console.log(POPULATION[i || 0].genome.debugString);
//	console.log(debug === POPULATION[i || 0].genome.debugString);
}

REUSE_GA.trainEvolution = function(data) {
	DEFAULT_TRANSFER_SET.disconnectTransfers();
	
	this.killR = this.killR ? this.killR
			: Math.min(1, Math.max(0, el('killRate').value / 100));
	this.mutation = this.mutation ? this.mutation
			: { wgt: Math.min(1, Math.max(0, el('mutationRateW').value / 100)),
				io: Math.min(1, Math.max(0, el('mutationRateN').value / 100)),
				size: Math.min(1, Math.max(0, el('mutationRateL').value / 100)) }
	
	var ga = this;
	var newChildFn = function(parent) {
		return ga.createChildGenome(parent.genome, ga.mutation.wgt, ga.mutation.io, ga.mutation.size);
	};
	
	evolve(data, this.killR, this.fitnessFunction, newChildFn, this.createRandomGenome);
	
}

REUSE_GA.createChildGenome = function(parentGenome, wgtMutation, ioMutation, sizeMutation) {
	var child = new TransferSet(), sizeD = dMut(sizeMutation);
	n = parentGenome.transfers.length + Math.min(0, sizeD);
	
	for (var i = 0; i < n; i++) {
		var ioP = pMut(ioMutation), wgtR = rMut(wgtMutation), pt = parentGenome.transfers[i];
		if (ioP) randomTransfer(DISPLAY.anns[pt.inAnnId], DISPLAY.anns[pt.outAnnId], child);
		else child.addTransfer(pt.copy());
		child.transfers[child.transfers.length-1].multiplyWeight(Math.exp(wgtR));
	}
	if (sizeD > 0) randomTransfer(DISPLAY.anns[FOCUS], getLenders(), child); // random lender?
	if (!sanityCheck(child, parentGenome)) console.log("PROBLEMO");
	return child;
}

function sanityCheck(ts1, ts2) {
	for (var i = 0; i < ts1.transfers.length; i++) {
		if (contains(ts1.transfers[i], ts2.transfers)) return false;
	}
	for (var i = 0; i < ts2.transfers.length; i++) {
		if (contains(ts2.transfers[i], ts1.transfers)) return false;
	}
	return true;
}

REUSE_GA.createRandomGenome = function() {
	var max = 10, min = 2;
	size = min + Math.round(Math.random() * (max - min));
	return randomTransferSet(size);
}

function randomTransferSet(size) {
	var borrower = DISPLAY.anns[FOCUS], lenders = getLenders(), ts = new TransferSet();
	for (var i = 0; i < size; i++) randomTransfer(borrower, lenders, ts);
	return ts;
}

function randomTransfer(borrower, lenders, ts) {
	var inVsOut = Math.random() < 0.5, lender = lenders.length ? oneOf(lenders) : lenders;
	if (inVsOut) connectNodes(ts, lender, 1 + randomInt(lender.numLayers(true) - 1), null,
			borrower, randomInt(borrower.numLayers(true)), null);
	else connectNodes(ts, borrower, randomInt(borrower.numLayers(true) - 1), null,
			lender, randomInt(lender.numLayers(true)), null);
}

function getLenders() {
	var lenders = [];
	for (var i = 0; i < DISPLAY.anns.length; i++) if (i != FOCUS) lenders.push(DISPLAY.anns[i]);
	return lenders;
}