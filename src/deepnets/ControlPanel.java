package deepnets;

import java.util.Collection;

public class ControlPanel {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		for (double i = -2; i < 2.1; i += 0.1) {
//			System.out.println(Utils.gaussianProbLE(i, 1));
			
			System.out.println(Utils.erf(i));
			
		}
		
	}
	/**
	 * lRate init and final learn rate
	 * mRate init and final momentum
	 * sRate init and final stochasticity
	 * @param inputNodes
	 * @param outputNodes
	 * @param data
	 * @param epochs
	 * @param lRate0
	 * @param lRateF
	 * @param mRate0
	 * @param mRateF
	 * @param sRate0
	 * @param sRateF
	 */
	public static void learnFromBackPropagation(Collection<? extends Node> inputNodes, Collection<? extends Node> outputNodes,
			Collection<DataPoint> data, int epochs,
			double lRate0, double lRateF, double mRate0, double mRateF, double sRate0, double sRateF) {
		double lRate = lRate0;
		double mRate = mRate0;
		double sRate = sRate0;
		double lastError = 1;
		for (int i = 0; i < epochs; i++) {
			// stochasticity should correlate with (late-epoch) error
			for (DataPoint dp : data) {
				FFNeuralNetwork.feedForward(inputNodes, dp.getInputs());
				FFNeuralNetwork.backPropagate(outputNodes, lRate, mRate, sRate, dp.getOutputs());
			}
			final double stErr = FFNeuralNetwork.stdError(inputNodes, outputNodes, data);
			lRate = Utils.between(i, epochs, lRate0, lRateF);
			mRate = Utils.between(i, epochs, mRate0, mRateF);
			if (i % 10 == 0 && stErr > 0.35) {
				sRate = Utils.between(i, epochs, sRate0, sRateF) * (1 - (lastError - stErr)) * stErr;
//				System.out.println(sRate + "	"+stErr);
				lastError = stErr;
			}
			else sRate = 0;
		}
	}
	
}
