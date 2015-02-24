package reasoner;

import java.util.Collection;

import modeler.*;
import deepnets.*;

public class Foresight {

	public static double[] getBestPredictedNextState(ModelLearner modeler, double[] initialStateVars,
			int numRuns, int jointAdjustments) {
		double[] mc = montecarlo(modeler, initialStateVars, null, 1, numRuns, jointAdjustments);
		return deterministicRoundingUnnormalized(mc); // probabilistic?
	}
	public static double[] montecarlo(ModelLearner modeler, double[] initialStateVars, double[] actionVars,
			int numSteps, int numRuns, int jointAdjustments) {
		return montecarlo(modeler, initialStateVars, actionVars, null, numSteps, numRuns, jointAdjustments, 0);
	}
	public static double[] montecarlo(ModelLearner modeler, double[] initialStateVars, double[] actionVars,
			int numSteps, int numRuns, int jointAdjustments, double skewFactor) {
		return montecarlo(modeler, initialStateVars, actionVars, null, numSteps, numRuns, jointAdjustments, skewFactor);
	}
	public static double[] montecarlo(ModelLearner modeler, double[] initialStateVars, double[] actionVars,
			MultiRewardAssessment mra, int numSteps, int numRuns, int jointAdjustments, double skewFactor) {
		double[] result = new double[initialStateVars.length];
		if (actionVars != null) modeler.observeAction(actionVars);
		double totalRealism = 0;
		for (int r = 0; r < numRuns; r++) {
			double[] stateVars = new double[initialStateVars.length];
			System.arraycopy(initialStateVars, 0, stateVars, 0, initialStateVars.length);
			double realism = 0;
			for (int i = 0; i < numSteps; i++) {
				modeler.observePreState(stateVars);
				modeler.feedForward();
				int j = 0;
				Collection<? extends Node> outputs = modeler.getTransitionsModule().getNeuralNetwork().getOutputNodes();
				double[] newStateVars = new double[stateVars.length];
				for (Node n : outputs) newStateVars[j++] = n.getActivation();
				double[] allVars = ModelLearnerHeavy.concatVars(stateVars, actionVars, newStateVars);
				newStateVars = modeler.upJointOutput(allVars, allVars.length - newStateVars.length, jointAdjustments);
//				newStateVars = modeler.upFamiliarity(allVars, newStateVars.length, jointAdjustments,
//						jointAdjustments, 2.5, .5, 0, 0.1); // <- hacky!
				realism = 1; //modeler.getFamiliarity(allVars); // assumes same action held
				if (mra != null) mra.observeState(i, newStateVars, realism);
				stateVars = newStateVars;
			}
			for (int j = 0; j < result.length; j++) result[j] += stateVars[j] * realism;
			totalRealism += realism;
		}
		for (int j = 0; j < result.length; j++) result[j] /= totalRealism;
		return result;
	}
	
	private static double estimateRealism(double[] stateVars, TransitionFamiliarityAssessor tra) {
		// TODO use auxiliary realism network (prob < thresh -> false)
		double t1 = 0; double t2 = 0; // HACK
		for (int i = 0; i < stateVars.length/2; i++) t1 += stateVars[i];
		for (int i = stateVars.length/2; i < stateVars.length; i++) t2 += stateVars[i];
		return t1 == 1 && t2 == 1 ? 1 : 0;
	}
	public static double[] deterministicRoundingUnnormalized(double[] in) {
		double[] out = new double[in.length];
		for (int i = 0; i < in.length; i++) out[i] = Math.round(in[i]);
		return out;
	}
	public static double[] probabilisticRoundingUnnormalized(double[] in) {
		double[] out = new double[in.length];
		for (int i = 0; i < in.length; i++) out[i] = Math.random() < in[i] ? 1 : 0;
		return out;
	}
	public static double[] probabilisticRoundingNormalized(double[] in) {
		double sum = 0;
		for (int i = 0; i < in.length; i++) sum += in[i];
		double[] out = new double[in.length];
		for (int i = 0; i < in.length; i++) out[i] = Math.random() < in[i] / sum ? 1 : 0;
		return out;
	}
	public static double[] probabilitySkewing(double[] in, double factor) { // e.g. factor= .05, 0.1
		if (factor <= 0.005) return in;
		double[] out = new double[in.length];
		for (int i = 0; i < in.length; i++) out[i] = probabilitySkewing(in[i], factor);
		return out;
	}
	public static double probabilitySkewing(double in, double factor) { // e.g. factor= .05, 0.1
		return Math.min(1, Math.max(0, in * (1 + factor) - factor / 2));
	}
	
}
