package modularization;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ann.Connection;
import ann.FFNeuralNetwork;
import ann.Node;
import ann.Utils;

public class SoftWeightSharing {

	private static final double wedgeBase = 0.01;
	protected final ShareWeightCluster[] clusters;
	private final int learnInterval;
	private final double learningRate;
	private final double momentum;
	private final double wgtDecay;
	private Set<Connection> connections = new HashSet<Connection>();
	private int t;

	public SoftWeightSharing(int numClusters, int learnInterval,
			double learningRate, double momentum, double wgtDecay) {
		this.clusters = new ShareWeightCluster[numClusters];
		this.learnInterval = learnInterval;
		this.learningRate = learningRate;
		this.momentum = momentum;
		this.wgtDecay = wgtDecay;
	}
	
	public void backPropagate(Collection<? extends Node> nodes, double[] targets) {
		if (connections.isEmpty()) {
			loadConnections(nodes);
			initializeClusters();
		}
		if (++t % learnInterval == 0) {
			Map<Connection, double[]> r = updateWeights();
			updateClusters(r);
		}
		
		FFNeuralNetwork.backPropagate(nodes, learningRate, momentum, wgtDecay, targets);
	}
	
	private Map<Connection, double[]> updateWeights() {
		Map<Connection, double[]> r = new HashMap<Connection, double[]>();
		for (Connection conn : connections) {
			final double w = conn.getWeight().getWeight();
			final double[] cr = new double[clusters.length];
			double delta = calcWeightChange(w, cr);
			conn.getWeight().chgWeight(delta * learningRate * 0.1);
			r.put(conn, cr);
		}
		
		return r;
	}
	
	protected void updateClusters(Map<Connection, double[]> rMap) {
		int i = 0;
		for (ShareWeightCluster cluster : clusters) {
			double dMean = 0;
			double dStDev = 0;
			double dProportion = 0;
			for (Connection conn : connections) {
				final double w = conn.getWeight().getWeight();
				final double r = rMap.get(conn)[i];
				final double x = cluster.getMean() - w;
				final double v = cluster.getVariance();
				dMean += r * x / v;
				dStDev -= r * (x * x - v) / (v * cluster.getStdev());
				dProportion += 1 - r / cluster.getProportion();
			}
			cluster.update(dMean, dStDev, dProportion);
			i++;
		}
	}
	
	private double calcR(double w, ShareWeightCluster c, double denom) {
		return c.getProportion() * pdf(w, c.getMean(), c.getStdev()) / denom;
	}
	
	private double calcDenom(double w) {
		double sum = 0;
		for (ShareWeightCluster c : clusters) {
			sum += c.getProportion() * pdf(w, c.getMean(), c.getStdev());
		}
		return sum;
	}
	
	private double pdf(double x, double mu, double sigma) {
		return Utils.wedgie(x, mu, sigma, wedgeBase);
//		return Utils.gaussianPdf(x, mu, sigma);
	}
	
	protected double calcWeightChange(double w, double[] cr) {
		return stochasticWgtChg(w, cr);
//		return derivativeWgtChg(w, cr);
	}
	
	protected double stochasticWgtChg(double w, double[] cr) {
		final double denom = calcDenom(w);
		int i = 0;
		for (ShareWeightCluster c : clusters) {
			final double p = calcR(w, c, denom);
			cr[i++] = p;
		}
		// choose belonging cluster randomly based on belonging ratio
		ShareWeightCluster c = clusters[Utils.wheelOfFortuneDenomed(cr)];
		double h = 1.0 / learnInterval;
		double g = h * Math.random() + (1 - h) * 1;
		double newWgt = Utils.between(g, 1, w, c.getMean());
		return newWgt - w;
	}
	

	
	protected double derivativeWgtChg(double w, double[] cr) {
		final double denom = calcDenom(w);
		double delta = 0;
		int i = 0;
		for (ShareWeightCluster c : clusters) {
			final double p = calcR(w, c, denom);
			cr[i++] = p;
			double d = Utils.dWedgie(w, c.getMean(), c.getStdev(), wedgeBase);
			delta -= p * d;
//			delta -= p * (c.getMean() - w) / c.getVariance();
			if (!Double.isFinite(delta)) {
				System.out.println();
			}
		}
		return delta;
	}
	
	protected void initializeClusters() {
		double mean = 0;
		double stdev = 0.2;
		double p = 1.0 / clusters.length;
		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new ShareWeightCluster(mean, stdev, p);
		}
		// TODO draw means and stdevs from distribution of initial connection weights
	}

	private void loadConnections(Collection<? extends Node> nodes) {
		for (Node n : nodes) loadConnections(n);
	}

	private void loadConnections(Node node) {
		Collection<Connection> inputConnections = node.getInputConnections();
		for (Connection conn : inputConnections) {
			connections.add(conn);
			loadConnections(conn.getInputNode());
		}
	}

	public double getLearningRate() {
		return learningRate;
	}

	public double getMomentum() {
		return momentum;
	}

	public double getWgtDecay() {
		return wgtDecay;
	}
}
