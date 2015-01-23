
var NUM_ACTIONS = 2;

var MEMORY_A_SIZE;
var MEMORY_A = [];
var MEMORY_B_SIZE;
var MEMORY_B = [];
var VALID_INPUTS = [el("button1").innerHTML, el("button2").innerHTML];
var NEW_GUESS = null;
var NEW_GUESS_I = null;

var T = 0;
var aiWin = 0;

function observe(value, reportEl) {
	if (!contains(value, VALID_INPUTS)) {
		reportEl.innerHTML = "invalid input";
		return;
	}
	if (MEMORY_A.length >= MEMORY_A_SIZE) MEMORY_A.splice(MEMORY_A_SIZE-1,1);
	if (MEMORY_B.length >= MEMORY_B_SIZE) MEMORY_B.splice(MEMORY_B_SIZE-1,1);
	var userOutput = [], userI = 0;
	for (var i = 0; i < NUM_ACTIONS; i++) {
		if (value == VALID_INPUTS[i]) {
			userI = i;
			userOutput[i] = 1;
		} else {userOutput[i] = 0;}
	}
	
	evaluate(userI);
	learnFromObservation(userOutput);
	
	MEMORY_A.splice(0,0,userOutput);
	MEMORY_B.splice(0,0,aiWin?1:0);
	MEMORY_B[MEMORY_B.length-1] = GRAPH.lastResult / 100;
	reportEl.innerHTML = MEMORY_A + " : " + MEMORY_B;
	
	guessAgain();
}

var aiWinsEl = el("aiWins");
var playerWinsEl = el("playerWins");

function evaluate(userI) {
	aiWin = NEW_GUESS_I == userI;
	if (aiWin) aiWinsEl.innerHTML = ++AI_WINS;
	else playerWins.innerHTML = ++PLAYER_WINS;
	GRAPH.addPoint(T++, (aiWin ? 1 : 0));
	GRAPH.redraw();
}

function learnFromObservation(output) {
	feedBack(ann1(), output);
}

function guessAgain() {
	var ann = ann1();
	ann.feedForward(MEMORY_A.concat(MEMORY_B));
	NEW_GUESS = newGuessFromOutput(ann.outputs);
	DISPLAY.redraw();
}

function newGuessFromOutput(outputs) {
	var max = -1;
	for (var i = 0; i < NUM_ACTIONS; i++) {
		var a = outputs[i].activation;
		if (a > max) {
			max = a; NEW_GUESS_I = i;
		}
	}
	var result = [];
	for (var i = 0; i < NUM_ACTIONS; i++) result[i] = 0;
	result[NEW_GUESS_I] = 1;
	return result;
}