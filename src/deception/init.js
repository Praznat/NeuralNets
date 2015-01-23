

var DISPLAY = new ANNDisplay(el("displayDiv"));
function initANN() {
	MEMORY_A_SIZE = el('memA').value*NUM_ACTIONS;
	MEMORY_B_SIZE = el('memB').value*1;
	DISPLAY.setANN();
	pretrain(ann1(), 1000);
}
initANN();
function ann1() {return DISPLAY.anns[FOCUS]};

var REPORT_EL = el('inputFeedback');

var PLAYER_WINS = 0;
var AI_WINS = 0;

document.onkeypress = function(event) {inputty(String.fromCharCode(event.keyCode).toUpperCase())};
var b1 = el("button1"), b2 = el("button2");
b1.onclick = function(event) {inputty(b1.innerHTML)};
b2.onclick = function(event) {inputty(b2.innerHTML)};

el("resetButt").onclick = initANN;

var GRAPH = new Graph("AI Win Rate",el("performanceStatsDiv"),10)

function inputty(chary) {
	observe(chary, REPORT_EL);
}
