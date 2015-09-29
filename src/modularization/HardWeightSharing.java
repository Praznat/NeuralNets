package modularization;

import java.util.Map;

import ann.Connection;
import ann.Utils;

public class HardWeightSharing extends SoftWeightSharing {

	private double[] clusterMeans;

	public HardWeightSharing(double[] clusterMeans, int learnInterval,
			double learningRate, double momentum, double wgtDecay) {
		super(clusterMeans.length + 1, learnInterval, learningRate, momentum, wgtDecay);
		this.clusterMeans = clusterMeans;
	}
	
	@Override
	protected void initializeClusters() {
		double mean = Utils.mean(clusterMeans);
		double stdev = Utils.stdev(mean, clusterMeans);
		double p = 1.0 / clusters.length;
		for (int i = 0; i < clusters.length - 1; i++) {
			clusters[i] = new ShareWeightCluster(clusterMeans[i], stdev, p);
		}
		clusters[clusters.length - 1] = new ShareWeightCluster(mean, stdev * 4, p);
	}
	
	@Override
	protected void updateClusters(Map<Connection, double[]> rMap) {}

}
