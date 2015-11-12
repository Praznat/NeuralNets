package modeler;

import ann.*;
import utils.RandomUtils;

public class TransitionFamiliarityAssessor extends ModelNeuralNet {

	protected TransitionFamiliarityAssessor(ActivationFunction actFn, int[] numHidden, int errorHalfLife) {
		super(actFn, numHidden, errorHalfLife);
	}

	@Override
	protected void analyzeTransition(TransitionMemory tm, double lRate, double mRate, double sRate) {
		final double[] ins = tm.getAllVars();
		final double[] targets = {1};
		int times = (int)Math.sqrt(ins.length); //ins.length*10; // TODO should this be more??
		for (int i = 0; i < times; i++) {
			nnLearn(ins, targets, lRate, mRate, sRate);
			// suppress non-experienced outcomes
			final double[] suppressIns = RandomUtils.randomBits(ins.length);
			final int k = (int)(Math.random()*suppressIns.length);
			suppressIns[k] = 1 - ins[k]; // ensure different from in
			nnLearn(suppressIns, new double[] {0}, lRate, mRate, sRate);
		}
	}

}
