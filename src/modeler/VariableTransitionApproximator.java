package modeler;

import deepnets.*;

public class VariableTransitionApproximator extends ModelerModule {
	
	public VariableTransitionApproximator(ActivationFunction actFn, int[] numHidden, int errorHalfLife) {
		super(actFn, numHidden, errorHalfLife);
		if (actFn != ActivationFunction.SIGMOID0p5) throw new IllegalStateException("must use sigmoidal if representing probabilities");
	}
	
	@Override
	protected void analyzeTransition(TransitionMemory tm, double lRate, double mRate, double sRate) {
		final double[] ins = tm.getPreStateAndAction();
		final double[] targets = tm.getPostState();
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

}
