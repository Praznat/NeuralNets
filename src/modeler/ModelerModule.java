package modeler;

import java.util.*;

import utils.Decayer;
import deepnets.*;

public abstract class ModelerModule {

	private static final double NO_ERR_CHG_THRESH = 0.00000001;
	private static final int STRIKES = 15000000;
	
	private double errorEMA = 0;
	private int emaStep = 100;

	protected FFNeuralNetwork ann;
	private final ActivationFunction actFn;
	private final Decayer decayer;
	
	protected ModelerModule(ActivationFunction actFn, int[] numHidden, int errorHalfLife) {
		ann = new FFNeuralNetwork(actFn, 0, 0, numHidden);
		this.actFn = actFn;
		decayer = new Decayer(errorHalfLife);
	}
	
	protected void adjustNNSize(int necInSize, int necOutSize) {
		final int inShortage = necInSize - ann.getInputNodes().size();
		final int outShortage = necOutSize - ann.getOutputNodes().size();
		if (inShortage > 0) {
			for (int i = 0; i < inShortage; i++) ann.addNode(0, actFn, null);
		}
		if (outShortage > 0) {
			for (int i = 0; i < outShortage; i++) ann.addNode(ann.getNumLayers()-1, actFn,
					new AccruingWeight(1.0, false)); // "true" when you want standard output to be zero?
		}
	}
	
	public void learn(Collection<TransitionMemory> memories, double stopAtErrThreshold,
			long displayProgressMs, int iterations, double lRate, double mRate, double sRate,
			boolean isRecordingTraining, ArrayList<Double> trainingErrorLog) {
		long lastMs = System.currentTimeMillis();
		double lastErr = 1;
		int strikesLeft = STRIKES;
		for (int i = 0; i < iterations; i++) {
			for (TransitionMemory m : memories) analyzeTransition(m, lRate, mRate, sRate);
			double err = getConfidenceEstimate();
			if (stopAtErrThreshold > 0 && err < stopAtErrThreshold) break;
			if (displayProgressMs > 0 && System.currentTimeMillis() - lastMs >= displayProgressMs) {
				System.out.println(Utils.round(((double)i)*100 / iterations, 2) + "%"
						+ "	err:	" + err);
				lastMs = System.currentTimeMillis();
			}
			if (isRecordingTraining) trainingErrorLog.add(err);
			if (lastErr - err < NO_ERR_CHG_THRESH && strikesLeft-- <= 0) {
				System.out.println("Not learning anymore");
				break;
			}
			lastErr = err;
		}
	}
	
	protected abstract void analyzeTransition(TransitionMemory dp, double lRate, double mRate, double sRate);

	public FFNeuralNetwork getNeuralNetwork() {
		return ann;
	}
	
	protected void nnLearn(double[] ins, double[] targets, double lRate, double mRate, double sRate) {
		adjustNNSize(ins.length, targets.length);
		FFNeuralNetwork.feedForward(ann.getInputNodes(), ins);
		final double error = FFNeuralNetwork.getError(targets, ann.getOutputNodes());
		observeError(error);
		FFNeuralNetwork.backPropagate(ann.getOutputNodes(), lRate, mRate, sRate, targets);
		
		// print out I/O for debugging
//		String s = "";
//		for (double d : ins) s += d+"	";
//		s += "	:	";
//		for (Node n : ann.getOutputNodes()) s += n.getActivation()+"	";
//		System.out.println(s);
	}
	
	protected void observeError(double error) {
		errorEMA = decayer.newEMA(error, errorEMA, ++emaStep);
	}
	
	public double getConfidenceEstimate() {
		return errorEMA;
	}

	public double[] getOutputActivations() {
		Collection<? extends Node> nodes = this.getNeuralNetwork().getOutputNodes();
		double[] result = new double[nodes.size()];
		int i = 0;
		for (Node n : nodes) result[i++] = n.getActivation();
		return result;
	}
	
	public void setANN(FFNeuralNetwork ann) {
		this.ann = ann;
	}

}
