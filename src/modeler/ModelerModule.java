package modeler;

import utils.Decayer;
import deepnets.*;

public abstract class ModelerModule {

	private double errorEMA = 0;
	private int emaStep = 100;

	protected final FFNeuralNetwork ann;
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
					new AccruingWeight(1.0));
		}
	}
	
	protected abstract void analyzeTransition(TransitionMemory dp, double lRate, double mRate, double sRate);

	public FFNeuralNetwork getNeuralNetwork() {
		return ann;
	}
	
	protected void observeError(double error) {
		errorEMA = decayer.newEMA(error, errorEMA, ++emaStep);
	}
	
	public double getConfidenceEstimate() {
		return errorEMA;
	}

}
