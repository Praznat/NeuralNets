var TOOMANYPOINTS = 100;

window.Graph = function(title, parentElement, halfLife) {
	this.title = title;
	this.parent = parentElement;
	this.canvas = document.createElement('canvas');
	this.ctx = this.canvas.getContext('2d');
	var deviceRatio = window.devicePixelRatio || 1;
	var storeRatio = (this.ctx.webkitBackingStorePixelRatio ||
			this.ctx.mozBackingStorePixelRatio ||
			this.ctx.msBackingStorePixelRatio ||
			this.ctx.oBackingStorePixelRatio ||
			this.ctx.backingStorePixelRatio || 1);
	this.pixelMult = deviceRatio / storeRatio;
	this.display = true;

	parentElement.appendChild(this.canvas);
	
	this.clear();
	this.halfLife = halfLife;
	this.ema = 0;
	
	var dis = this;
	this.canvas.addEventListener('click', function(event) {
		dis.display = !dis.display;
	}, false);
	
	this.lastResult = 0;
}

Graph.prototype.clear = function() {
	this.points = [];
	this.maxY = -1999999999999;
	this.minY = 1999999999999;
	this.maxX = -1999999999999;
	this.minX = 1999999999999;
}

Graph.prototype.addPoint = function(x, y) {
	if (this.points.length >= TOOMANYPOINTS) {
		this.points.splice(0,1);
		this.minX = this.points[0][0]; // hacky
	}
	var decay = decayFactor(1, this.halfLife);
	this.ema = this.ema * decay + y * (1 - decay);
	this.points.push([x, this.ema]);
	if (x > this.maxX) this.maxX = x;
	if (x < this.minX) this.minX = x;
	if (y > this.maxY) this.maxY = y;
	if (y < this.minY) this.minY = y;
}

Graph.prototype.redraw = function() {
	if (!this.display || !this.points.length) return;
	var q = this.canvas, ctx = this.ctx;
	var w = q.width = this.parent.offsetWidth * this.pixelMult;
	var h = q.height = this.parent.offsetHeight * this.pixelMult;
	q.style.width = this.parent.offsetWidth + 'px';
	q.style.height = this.parent.offsetHeight + 'px';

	ctx.beginPath();
	ctx.moveTo(0,this.yToPx(0, h));
	ctx.strokeStyle=BLACK;
	for (var i = 0; i < this.points.length; i++) {
		var p = this.points[i], x = p[0], y = p[1];
		var ax = this.xToPx(x, w), ay = this.yToPx(y, h);
//		ctx.moveTo(ax+1, ay);
		ctx.lineTo(ax, ay);
		ctx.stroke();
	}
	ctx.closePath();
	
	drawText(ctx, this.title, w / 3, h / 8);
	var lastP = this.points[this.points.length-1];
	this.lastResult = Math.round(lastP[1] * 10000)/100;
	drawText(ctx, this.lastResult+"%", 5 * w / 8, h / 8);
	drawText(ctx, "turn : "+lastP[0], 3 * w / 8, h*7 / 8);
}

function drawText(ctx, txt, x, y) {
	ctx.fillStyle = WHITE;
	ctx.fillText(txt, x, y);
	ctx.strokeStyle = BLACK;
	ctx.strokeText(txt, x, y);
}

Graph.prototype.xToPx = function(x, w) {
	return w * (x - this.minX) / (this.maxX - this.minX);
}
Graph.prototype.yToPx = function(y, h) {
	return h * (this.maxY - y) / (this.maxY - this.minY)
}