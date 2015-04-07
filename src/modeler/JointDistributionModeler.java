package modeler;

import deepnets.ActivationFunction;

public class JointDistributionModeler extends ModelerModule {

	protected JointDistributionModeler(ActivationFunction actFn, int[] numHidden, int errorHalfLife) {
		super(actFn, numHidden, errorHalfLife);
	}

	@Override
	protected void analyzeTransition(TransitionMemory tm, double lRate,
			double mRate, double sRate) {
		final double[] ins = tm.getAllVars();
		final double[] postState = tm.getPostState();
		final double[] targets = new double[postState.length + 1];
		System.arraycopy(postState, 0, targets, 0, postState.length);
		targets[postState.length] = 1;
		nnLearn(ins, targets, lRate, mRate, sRate);
	}

	@Deprecated
	protected void analyzeTransitionOLD(TransitionMemory tm, double lRate,
			double mRate, double sRate) {
		final double[] ins = tm.getPostState();
		final double[] targets = tm.getPostState();
		nnLearn(ins, targets, lRate, mRate, sRate);
	}

}
