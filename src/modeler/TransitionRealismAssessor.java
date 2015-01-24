package modeler;

import deepnets.*;

public class TransitionRealismAssessor extends ModelerModule {

	protected TransitionRealismAssessor(ActivationFunction actFn, int[] numHidden, int errorHalfLife) {
		super(actFn, numHidden, errorHalfLife);
	}

	@Override
	protected void analyzeTransition(TransitionMemory tm, double lRate, double mRate, double sRate) {
		final double[] ins = tm.getAllVars();
		final double[] targets = {1};
		adjustNNSize(ins.length, targets.length);
		FFNeuralNetwork.feedForward(ann.getInputNodes(), ins);
		final double error = FFNeuralNetwork.getError(targets, ann.getOutputNodes());
		observeError(error);
		FFNeuralNetwork.backPropagate(ann.getOutputNodes(), lRate, mRate, sRate, targets);
	}


}
