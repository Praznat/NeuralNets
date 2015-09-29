package modeler;

import java.util.*;

import ann.*;
import modularization.SoftWeightSharing;
import utils.Decayer;

public abstract class ModelerModule {

	private static final double NO_ERR_CHG_THRESH = 0.00000001;
	private static final int STRIKES = 15000000;
	
	private double errorEMA = 0;
	private int emaStep = 100;

	protected FFNeuralNetwork ann;
	private final ActivationFunction actFn;
	private final Decayer decayer;
	
	private SoftWeightSharing wgtSharer;
	
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
		long lastMs = 0;
		double lastErr = 1;
		int strikesLeft = STRIKES;
		for (int i = 0; i < iterations; i++) {

			Collections.shuffle((List<TransitionMemory>)memories);
			for (TransitionMemory m : memories) analyzeTransition(m, lRate, mRate, sRate);
			double err = getConfidenceEstimate();
			if (displayProgressMs > 0 && System.currentTimeMillis() - lastMs >= displayProgressMs) {
				System.out.println(Utils.round(((double)i)*100 / iterations, 2) + "%"
						+ "	err:	" + err);
				lastMs = System.currentTimeMillis();
			}
			if (stopAtErrThreshold > 0 && err < stopAtErrThreshold) break;
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
	
	private void setupLearn(double[] ins, double[] targets, double lRate, double mRate, double sRate) {
		adjustNNSize(ins.length, targets.length);
		FFNeuralNetwork.feedForward(ann.getInputNodes(), ins);
		final double error = FFNeuralNetwork.getError(targets, ann.getOutputNodes());
		observeError(error);	
	}
	
	protected void nnLearn(double[] ins, double[] targets, double lRate, double mRate, double sRate) {
		setupLearn(ins, targets, lRate, mRate, sRate);
		if (wgtSharer != null) wgtSharer.backPropagate(ann.getOutputNodes(), targets);
		else FFNeuralNetwork.backPropagate(ann.getOutputNodes(), lRate, mRate, sRate, targets);
		
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

	public void setWgtSharer(SoftWeightSharing wgtSharer) {
		this.wgtSharer = wgtSharer;
	}

}
