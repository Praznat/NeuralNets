package modularization;

public class ShareWeightCluster {
	private double mean;
	private double stdev;
	private double proportion;
	
	public ShareWeightCluster(double mean, double stdev, double proportion) {
		this.mean = mean;
		this.stdev = stdev;
		this.proportion = proportion;
	}

	public double getMean() {
		return mean;
	}

	public double getStdev() {
		return stdev;
	}
	public double getVariance() {
		return stdev * stdev;
	}

	public double getProportion() {
		return proportion;
	}

	public void update(double dMean, double dStDev, double dProportion) {
		this.mean += dMean;
		this.stdev += dStDev;
		this.proportion += dProportion;
	}
}
