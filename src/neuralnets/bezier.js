function createBezier(a1, a2, F, N) {
	var points = [a1.getXY(), [], [], a2.getXY()],
	dist = distance(points[0], points[3]),
	mp1 = [points[0][0] - (points[3][1]-points[0][1]), points[0][1] + (points[3][0]-points[0][0])],
	mp2 = [points[3][0] - (points[3][1]-points[0][1]), points[3][1] + (points[3][0]-points[0][0])],
	sidedist1 = distance(points[3], mp1),
	sidedist2 = distance(points[0], mp2),
	dist1 = Math.min(dist, sidedist1) * F,
	dist2 = Math.min(dist, sidedist2) * F;
	points[1][0] = points[0][0] + Math.round(dist1 * a1.cosine());
	points[1][1] = points[0][1] + Math.round(dist1 * a1.sine());
	points[2][0] = points[3][0] + Math.round(dist2 * a2.cosine());
	points[2][1] = points[3][1] + Math.round(dist2 * a2.sine());
	var result = new GBezier(points, N);
	result.map();
	return result;
}
window.GBezier = function(list, n) {
	this.points = list;
	this.intervals = n;
	this.map();
}
GBezier.prototype.map = function() {
	/* double[][][] V = new double[points.length-1][intervals+1][2]; */
	var V = [], n = this.intervals, pts = this.points;
	for (var c = 0; c < pts.length - 1; c++) {
		var v = [];
		for (var i = 0; i < n + 1; i++) {
			v.push([(1 - i/n)*pts[c][0] + (i/n)*pts[c+1][0],
			        (1 - i/n)*pts[c][1] + (i/n)*pts[c+1][1]]);
		}
		V[c] = v;
	}
	while (V.length > 1) {
		/* double[][][] VV = new double[V.length - 1][intervals+1][2]; */
		var VV = [];
		for (var c = 0; c < V.length - 1; c++) {
			var vv = [];
			for (var i = 0; i < n + 1; i++) {
				vv.push([(1 - i/n)*V[c][i][0] + (i/n)*V[c+1][i][0],
				         (1 - i/n)*V[c][i][1] + (i/n)*V[c+1][i][1]]);
			}
			VV[c] = vv;
		}
		V = VV;
	}
	/* XY = new int[2][V[0].length]; */
	this.XY = [[],[]];
	for(var i = 0; i < n + 1; i++) {
		this.XY[0][i] = Math.round(V[0][i][0]);
		this.XY[1][i] = Math.round(V[0][i][1]);
	}
}
GBezier.prototype.startPoint = function() {return this.points[0];}
GBezier.prototype.endPoint = function() {return this.points[this.points.length-1];}
GBezier.prototype.getPoint = function(p) {return this.points[p];}
GBezier.prototype.getPoint = function(p) {return this.points[p];}
GBezier.prototype.setPointX = function(x, i) {
	this.points[i][0] = x; this.map();
}
GBezier.prototype.setPointY = function(y, i) {
	this.points[i][1] = y; this.map();
}
GBezier.prototype.startArrow = function() {
	var p0 = this.points[0], p1 = this.points[1];
	return new Arrow(p0, p0[0] - p1[0], p0[1] - p1[1]);
}
GBezier.prototype.endArrow = function() {
	var p0 = this.points[this.points.length - 1], p1 = this.points[this.points.length - 2];
	return new Arrow(p0, p0[0] - p1[0], p0[1] - p1[1]);
}


window.Arrow = function(XY, dX, dY) {this.xy = XY; this.dx = dX; this.dy = dY;}
Arrow.prototype.getXY = function() {return this.xy;}
Arrow.prototype.hypoteneuse = function() {return Math.sqrt(this.dx*this.dx + this.dy*this.dy);}
Arrow.prototype.slope = function() {return this.dy / this.dx;}
Arrow.prototype.asine = function() {return Math.sin(Math.atan(this.slope()));}
Arrow.prototype.acosine = function() {return Math.cos(Math.atan(this.slope()));}
Arrow.prototype.sine = function() {return this.dy / this.hypoteneuse();}
Arrow.prototype.cosine = function() {return this.dx / this.hypoteneuse();}

