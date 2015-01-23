package modeler;

import deepnets.ActivationFunction.Gaussian;

public abstract class EnvTranslator {

	public abstract double[] toNN(double... n);

	public abstract double[] fromNN(double[] d);

	/** runs continuous variables into an array of radial basis functions (RBFs) to represent value 
	 * as distribution over various RBF centers 
	 * please use mins > actual variable mins and maxes < actual variable maxes*/
	public static EnvTranslator rbfEnvTranslator(final double[] mins, final double[] maxes, final int[] numRBFs) {
		return rbfEnvTranslator(mins, maxes, numRBFs, 0.5); // overlap must < 1
	}
	public static EnvTranslator rbfEnvTranslator(final double[] mins, final double[] maxes, final int[] numRBFs, final double overlap) {
		if (mins.length != maxes.length || mins.length != numRBFs.length)
			throw new IllegalStateException("mins size must equal maxes size");
		final Gaussian[][] rbfMatrix = new Gaussian[mins.length][];
		final double[][] obs = new double[mins.length][]; // just for testing
		int rbfn = 0;
		for (int i = 0; i < rbfMatrix.length; i++) {
			int nrbf = numRBFs[i];
			final Gaussian[] rbfs = new Gaussian[nrbf];
			obs[i] = new double[nrbf];
			rbfn += nrbf;
			final double min = mins[i];
			final double space = (maxes[i] - min) / (nrbf - 1);
			final double sharpness = Math.sqrt(-Math.log(overlap)) * 2 / space;
			for (int j = 0; j < nrbf; j++) rbfs[j] = new Gaussian(min + j * space, sharpness);
			rbfMatrix[i] = rbfs;
		}
		final int totalNumRBFs = rbfn;
		return new EnvTranslator() {
			public final double[][] observations = obs;
			@Override
			public double[] toNN(double... vals) {
				if (vals.length != mins.length) throw new IllegalStateException("vals size must equal mins/maxes size");
				double[] result = new double[totalNumRBFs];
				int k = 0;
				for (int i = 0; i < vals.length; i++) {
					final Gaussian[] rbfs = rbfMatrix[i];
					final double val = vals[i];
					for (int j = 0; j < rbfs.length; j++) {
						double w = rbfs[j].feed(val);
						result[k++] = w;
						observations[i][j] += w;
					}
				}
				return result;
			}
			@Override
			public double[] fromNN(double[] d) {
				double[] result = new double[mins.length];
				int past = 0;
				for (int i = 0; i < mins.length; i++) {
					final Gaussian[] rbfs = rbfMatrix[i];
					double denom = 0;
					double total = 0;
					for (int j = 0; j < rbfs.length; j++) {
						final double act = d[past + j];
						total += act * rbfs[j].getCenter();
						denom += act;
					}
					result[i] = total / denom;
					past += rbfs.length;
				}
				return result;
			}
			public double[] fromNNExact(double[] d) {
				double[] result = new double[mins.length];
				int past = 0;
				for (int i = 0; i < mins.length; i++) {
					final Gaussian[] rbfs = rbfMatrix[i];
					int g1 = -1, g2 = -1;
					double hi = -1;
					for (int j = 0; j < rbfs.length; j++) {
						double a = d[past + j];
						if (a > hi) {
							hi = a;
							g1 = j;
						} else if (g1 > -1) {
							g2 = j <= 1 ? 0 : (a > d[past + j - 2] ? j : j - 2);
							break;
						}
					}
					if (g1 == -1) throw new IllegalStateException("Something wrong.");
					if (g2 == -1) g2 = rbfs.length-1;
					double[] d1 = rbfs[g1].backfeed2(d[past + g1]);
					double[] d2 = rbfs[g2].backfeed2(d[past + g2]);
					double val = Math.abs(d1[0]-d2[1]) < Math.abs(d1[1]-d2[0]) ? (d1[0]+d2[1])/2 : (d1[1]+d2[0])/2;
					result[i] = val;
					past += rbfs.length;
				}
				return result;
			}
		};
	}


	@Deprecated
	public static EnvTranslator bucketEnvTranslator(final double[] mins, final double[] maxes, final int[] bucketSizes) {
		if (mins.length != maxes.length || mins.length != bucketSizes.length)
			throw new IllegalStateException("mins must equal maxes");
		return new EnvTranslator() {
			private int totalSize = 0;
			{
				for (int b : bucketSizes) totalSize += b; // move this to constructor
			}
			@Override
			public double[] toNN(double... vals) {
				double[] result = new double[totalSize];
				int varPlace = 0;
				for (int i = 0; i < mins.length; i++) {
					final int bucketSize = bucketSizes[i];
					final double min = mins[i], max = maxes[i], val = vals[i];
					final double space = (max - min) / bucketSize;
					final int place = (int) Math.min(bucketSize-1, Math.max(0, (val - min) / space))
							+ varPlace;
					result[place] = 1;
					varPlace += bucketSize;
				}
				return result;
			}
			@Override
			public double[] fromNN(double[] d) {
				double[] result = new double[mins.length];
				int varPlace = 0;
				for (int i = 0; i < mins.length; i++) {
					double sum = 0;
					double sumActivation = 0;
					final double min = mins[i];
					final int bucketSize = bucketSizes[i];
					final double space = (maxes[i] - min) / bucketSize;
					for (int j = 0; j < bucketSize; j++) {
						final int plc = varPlace + j;
						final double midVal = min + space * (j + 0.5);
						final double activation = d[plc];
						sum += activation * midVal;
						sumActivation += activation;
					}
					result[i] = sum / sumActivation;
					varPlace += bucketSize;
				}
				return result;
			}
		};
	}

}
