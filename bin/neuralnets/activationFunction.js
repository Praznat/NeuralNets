
window.ActivationFunction = function() {}

ActivationFunction.prototype.process = function(x) {}
ActivationFunction.prototype.derivative = function(x) {}

var GLOBAS_BIAS = 0.5;
var SIGMOID = new ActivationFunction();
SIGMOID.process = function(x) {
	return 1 / (1 + Math.exp(GLOBAS_BIAS - x));
}
SIGMOID.derivative = function(x) {
	return x*(1-x);
}

var RECTIFIER = new ActivationFunction();
RECTIFIER.process = function(x) {
	return Math.max(0, x);
}
RECTIFIER.derivative = function(x) {
	return x < 0 ? 0 : 1;
}