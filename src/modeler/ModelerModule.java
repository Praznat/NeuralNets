package modeler;

import java.util.Collection;

import utils.Decayer;
import deepnets.*;

public abstract class ModelerModule {

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
}
