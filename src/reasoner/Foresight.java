package reasoner;

import java.util.Collection;

import modeler.*;
import deepnets.*;

public class Foresight {

	public static double[] recurse(FFNeuralNetwork ffnn, double[] initialStateVars, int numSteps) {
		double[] stateVars = initialStateVars;
		for (int i = 0; i < numSteps; i++) {
			FFNeuralNetwork.feedForward(ffnn.getInputNodes(), stateVars);
			int j = 0;
			for (Node n : ffnn.getOutputNodes()) stateVars[j++] = n.getActivation();
		}
		return stateVars;
	}
	
	public static double[] montecarlo(ModelLearner modeler, double[] initialStateVars, double[] actionVars,
			int numSteps, int numRuns, double skewFactor) {
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
				Collection<? extends Node> outputs = modeler.getModelVTA().getNeuralNetwork().getOutputNodes();
				for (Node n : outputs) stateVars[j++] = n.getActivation();
				stateVars = probabilisticRounding(probabilitySkewing(stateVars, skewFactor));
				realism += estimateRealism(stateVars, modeler.getModelTRA());
			}
			for (int j = 0; j < result.length; j++) result[j] += stateVars[j] * realism;
			totalRealism += realism;
		}
		for (int j = 0; j < result.length; j++) result[j] /= totalRealism;
		return result;
	}
	
	private static double estimateRealism(double[] stateVars, TransitionRealismAssessor tra) {
		// TODO use auxiliary realism network (prob < thresh -> false)
		double t1 = 0; double t2 = 0; // HACK
		for (int i = 0; i < stateVars.length/2; i++) t1 += stateVars[i];
		for (int i = stateVars.length/2; i < stateVars.length; i++) t2 += stateVars[i];
		return t1 == 1 && t2 == 1 ? 1 : 0;
	}

	private static double[] probabilisticRounding(double[] in) {
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
