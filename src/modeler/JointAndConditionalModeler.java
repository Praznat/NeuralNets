package modeler;

import ann.*;

public class JointAndConditionalModeler extends ModelerModule {

	protected JointAndConditionalModeler(ActivationFunction actFn, int[] numHidden, int errorHalfLife) {
		super(actFn, numHidden, errorHalfLife);
		ann = new FFJointOutputNetwork(actFn, 0, 0, numHidden);
		if (actFn != ActivationFunction.SIGMOID0p5) throw new IllegalStateException("must use sigmoidal if representing probabilities");
	}
	
	@Override
	protected void analyzeTransition(TransitionMemory tm, double lRate, double mRate, double sRate) {
		final double[] ins = tm.getPreStateAndAction();
		final double[] targets = tm.getPostState();
		nnLearn(ins, targets, lRate, mRate, sRate);
	}


}
