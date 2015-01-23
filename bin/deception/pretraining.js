
window.PreTrainingSet = function(i, o) {
	this.inputs = i;
	this.output = o;
}

function hothand(size) {
	var r0= [], r1 = [];
	for (var i = 0; i < size; i++) {
		r0.push(0);
		r1.push(1);
	}
	return [new PreTrainingSet(r0, 0), new PreTrainingSet(r1, 1)];
}

function alternation(size) {
	var r0= [], r1 = [];
	for (var i = 0; i < size; i++) {
		r0.push(i % 2);
		r1.push((i+1) % 2);
	}
	return [new PreTrainingSet(r0, size % 2), new PreTrainingSet(r1, (size+1) % 2)];
}

function pretrain(ann, epochs) {
	var size = ann.inputs.length;
	var trainingData = hothand(size).concat(alternation(size));
	for (var t = 0; t < trainingData.length; t++) {
		for (var i = 0; i < epochs; i++) {
			var td = trainingData[t];
			ann.feedForward(td.inputs);
			feedBack(ann, [td.output]);
		}
	}
}
