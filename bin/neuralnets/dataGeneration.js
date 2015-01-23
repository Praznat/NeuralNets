

var SIG = {
		fn: function(x) {return 1 / (1 + Math.exp(-x));},
		desc: "SIG"
}
var SIN = {
		fn: function(x) {return (Math.sin(x) + 1) / 2;},
		desc: "SIN"
}
var COS = {
		fn: function(x) {return (Math.cos(x) + 1) / 2;},
		desc: "COS"
}
var RBF = {
		fn: function(x) {return Math.exp(-(x*x));},
		desc: "RBF"
}
var ID = {
		fn: function(x) {return x;},
		desc: ""
}
var INV = {
		fn: function(x) {return 1 - x;},
		desc: "NOT"
}
var COMBINER_A = {
		fn: function(f, g, x, y) {return f.fn(x) * g.fn(y);},
		desc: function(f, g, i, oldDesc) {return f.desc+"x"+i+"*"
			+ g.desc + (oldDesc.length ? "("+ oldDesc + ")" : "x0");}
}
var COMBINER_O = {
		fn: function(f, g, x, y) {return Math.min(1, f.fn(x) + g.fn(y));},
		desc: function(f, g, i, oldDesc) {return f.desc+"x"+i+"+" 
			+ g.desc + (oldDesc.length ? "(" + oldDesc + ")" : "x0");}
}

var FUNCTIONS = [ID, INV]; //SIG, SIN, COS, RBF, 
var COMBINERS = [COMBINER_A, COMBINER_O];

window.FunctionTree = function() {
		this.f = 0;
		this.i = 0;
		this.desc = '';
}
FunctionTree.prototype.restart = function() {this.f = 0; this.i = 0; this.desc = '';};
FunctionTree.prototype.calc = function(x, y, v) {
	var combiner = getInV(COMBINERS, v, this.f++),
	f1 = getInV(FUNCTIONS, v, this.f++),
	f2 = getInV(FUNCTIONS, v, this.f++);
	this.desc = combiner.desc(f1, f2, ++this.i, this.desc);
	return combiner.fn(f1, f2, x, y);
}
FunctionTree.prototype.calcV = function(x, y, v) {
	var combiner = getInV(COMBINERS, v, this.f++),
	f1 = getInV(FUNCTIONS, v, this.f++),
	f2 = getInV(FUNCTIONS, v, this.f++);
	this.desc = combiner.desc(f1, f2, ++this.i, this.desc);
	return combiner.fn(f1, f2, x, y);
}

function getInV(original, v, f) {
	var x = v[f % v.length];
	return original[x % original.length]
}

function randomV(size) {
	var result = [];
	for (var i = 0; i < size; i++) result.push(Math.floor(Math.random() * 100));
	return result;
}

function fillInputs(size) {
	var result = [], n = Math.pow(2, size);
	for (var i = 0; i < n; i++) {
		var dp = [], b = i.toString(2);
		for (var j = b.length; j < size; j++) dp.push('0');
		dp = dp.concat(b.split(''));
		result.push(dp.join(','));
	}
	return result;
}

function giveRandomOutputs(inputs, functionTree, v) {
	functionTree = functionTree || new FunctionTree();
	v = v || randomV(inputs[0].split(',').length);
	for (var i = 0; i < inputs.length; i++) {
		var input = inputs[i].split(','),
		output = input[0];
		functionTree.restart();
		for (var j = 1; j < input.length; j++) {
			output = functionTree.calc(input[j], output, v);
		}
		inputs[i] += ':' + Math.round(output);
	}
	console.log(functionTree.desc);
	console.log(inputs.join('\n'));
}





var CURR_L = null, INIT_L = null;
function addLogic(newLogic) {
	if (CURR_L != null) CURR_L.add(newLogic);
	else INIT_L = newLogic;
	CURR_L = newLogic;
	CURR_L.check();
}
function checkParent(logic) {
	if (!logic.parent) CURR_L = logic; //done;
	else logic.parent.check();
}
window.NotL = function(parent) {
	this.child = null;
	this.parent = parent;
}
NotL.prototype.add = function(logic) {
	this.child = logic;
}
NotL.prototype.check = function() {
	if (this.child) checkParent(this);
	else CURR_L = this;
}
NotL.prototype.eval = function(inputV) {
	return 1 - this.child.eval(inputV);
}
NotL.prototype.desc = function() {
	return "NOT[" + (this.child ? (this.child.desc() + "]") : "");
}
window.AndL = function(parent) {
	this.child1 = null;
	this.child2 = null;
	this.parent = parent;
}
AndL.prototype.add = function(logic) {
	if (!this.child1) this.child1 = logic;
	else this.child2 = logic;
}
AndL.prototype.check = function() {
	if (this.child2) checkParent(this);
	else CURR_L = this;
}
AndL.prototype.eval = function(inputV) {
	return this.child1.eval(inputV) * this.child2.eval(inputV);
}
AndL.prototype.desc = function() {
	return "AND[" + (this.child1 ? this.child1.desc() + (this.child2 ? ", " + this.child2.desc() + "]" : "") : "");
}
window.XL = function(parent) {
	this.parent = parent;
	this.id = xid;
}
XL.prototype.add = function(logic) {
	this.child = logic;
}
XL.prototype.check = function() {
	checkParent(this);
}
XL.prototype.eval = function(inputV) {
	return inputV[this.id];
}
XL.prototype.desc = function() {
	return "x<sub>"+this.id+"</sub>";
}

var unitsLeft = 1, xid = 0, maxId = 0;
el('rmNOT').onclick = function() {
	if (unitsLeft <= 0) return;
	addLogic(new NotL(CURR_L));
	var rmDisplay = el('rmDisplay');
	rmDisplay.innerHTML = INIT_L.desc();
}
el('rmAND').onclick = function() {
	if (unitsLeft <= 0) return;
	addLogic(new AndL(CURR_L));
	var rmDisplay = el('rmDisplay');
	unitsLeft++;
	rmDisplay.innerHTML = INIT_L.desc();
}
el('rmX').onclick = function() {
	if (unitsLeft <= 0) return;
	addLogic(new XL(CURR_L));
	var rmDisplay = el('rmDisplay');
	unitsLeft--;
	rmDisplay.innerHTML = INIT_L.desc();
	if (unitsLeft <= 0) rmDisplay.innerHTML += "<button onclick=rmDone()>DONE</button>";
}
el('rmInc').onclick = function() {
	el('rmi').innerHTML = ++xid;
	maxId = Math.max(maxId, xid);
}
el('rmDec').onclick = function() {
	xid = Math.max(0, xid-1);
	el('rmi').innerHTML = xid;
}

function rmDone() {
	var data = fillInputs(maxId+1);
	for (var i = 0; i < data.length; i++) {
		data[i] += ":" + INIT_L.eval(data[i].split(","));
	}
	el('trainingDataArea').value = data.join('\r\n');
	rmDisplay.innerHTML = '';
	INIT_L = null;
	CURR_L = null;
	unitsLeft = 1;
	xid = 0;
	maxId = 0;
	el('rmi').innerHTML = 0;
}
