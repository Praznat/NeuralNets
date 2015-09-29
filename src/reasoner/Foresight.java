package reasoner;

import java.util.*;

import ann.*;
import modeler.*;

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
	public static double[] montecarlo(ModelLearner modeler, double[] initialStateVars, double[] firstAction,
			List<double[]> actionChoices, int numSteps, int numRuns, int jointAdjustments, double skewFactor) {
		return montecarlo(modeler, initialStateVars, firstAction, actionChoices, null, numSteps, numRuns,
				jointAdjustments, skewFactor);
	}
	public static double[] montecarlo(ModelLearner modeler, double[] initialStateVars, double[] firstAction,
			List<double[]> actionChoices, MultiRewardAssessment mra, int numSteps, int numRuns,
			int jointAdjustments, double skewFactor) {
		modeler.clearWorkingMemory();
		double[] result = new double[initialStateVars.length];
		double totalRealism = 0;
		for (int r = 0; r < numRuns; r++) {
			double[] stateVars = new double[initialStateVars.length];
			System.arraycopy(initialStateVars, 0, stateVars, 0, initialStateVars.length);
			double realism = 0;
			for (int i = 0; i < numSteps; i++) {

				double[] action = i == 0 || actionChoices == null ? firstAction : actionChoices.get(r % actionChoices.size());
//				double[] action = i == 0 || actionChoices == null ? firstAction : RandomUtils.randomOf(actionChoices);
				modeler.observeAction(action);
				modeler.observePreState(stateVars);
				modeler.feedForward();
				// TODO move collections out of loop
				double[] newStateVars = newStateVars(modeler, stateVars, action, jointAdjustments);
//				newStateVars = modeler.upFamiliarity(allVars, newStateVars.length, jointAdjustments,
//						jointAdjustments, 2.5, .5, 0, 0.1); // <- hacky!
				realism = 1; //modeler.getFamiliarity(allVars); // assumes same action held
				if (mra != null) mra.observeState(i, realism, newStateVars);
				stateVars = probabilitySkewing(newStateVars, skewFactor);
			}
			for (int j = 0; j < result.length; j++) result[j] += stateVars[j] * realism;
			totalRealism += realism;
		}
		for (int j = 0; j < result.length; j++) result[j] /= totalRealism;
		modeler.clearWorkingMemory();
		return result;
	}
	public static boolean knowsWhatItKnows(ModelLearner modeler, double[] stateVars, double[] action,
			int jointAdjustments, double certaintyThreshold) {
		modeler.clearWorkingMemory();
		modeler.observeAction(action);
		modeler.observePreState(stateVars);
		modeler.feedForward();
		modeler.clearWorkingMemory();
		double[] newStateVars = newStateVars(modeler, stateVars, action, jointAdjustments);
		return estimateCertainty(newStateVars) > certaintyThreshold;
	}
	
	private static double[] newStateVars(ModelLearner modeler, double[] stateVars, double[] action, int jointAdjustments) {
		Collection<? extends Node> outputs = modeler.getTransitionsModule().getNeuralNetwork().getOutputNodes();
		double[] newStateVars = new double[stateVars.length];
		int j = 0;
		for (Node n : outputs) newStateVars[j++] = n.getActivation();
		double[] allVars = ModelLearnerHeavy.concatVars(stateVars, action, newStateVars);
		if (jointAdjustments > 0 && modeler.getFamiliarityModule() != null)
			newStateVars = modeler.upJointOutput(allVars, allVars.length - newStateVars.length, jointAdjustments);
		return newStateVars;
	}
	
	public static void terraIncognita(ModelLearner modeler, double[] initialStateVars, double[] firstAction,
			List<double[]> actionChoices, MultiRewardAssessment mra, int numSteps, int numRuns, int jointAdjustments, double skewFactor) {
		modeler.clearWorkingMemory();
		Collection<? extends Node> outputsVTA = modeler.getTransitionsModule().getNeuralNetwork().getOutputNodes();
		ArrayList<? extends Node> outputsJDM = modeler.getFamiliarityModule().getNeuralNetwork().getOutputNodes();
		if (outputsJDM.isEmpty()) {
			mra.observeState(0, 1, 0);
			return;
		}
		for (int r = 0; r < numRuns; r++) {
			double[] stateVars = new double[initialStateVars.length];
			System.arraycopy(initialStateVars, 0, stateVars, 0, initialStateVars.length);
			for (int i = 0; i < numSteps; i++) {
				double[] action = i == 0 ? firstAction : actionChoices.get(r % actionChoices.size());
//				double[] action = i == 0 ? firstAction : RandomUtils.randomOf(actionChoices);
				modeler.observeAction(action);
				modeler.observePreState(stateVars);
				modeler.feedForward();
				int j = 0;
				double[] newStateVars = new double[stateVars.length];
				for (Node n : outputsVTA) newStateVars[j++] = n.getActivation();
				double[] allVars = ModelLearnerHeavy.concatVars(stateVars, action, newStateVars);
//				final double familiarity1 = outputsJDM.get(outputsJDM.size()-1).getActivation();
//				final double familiarity2 = estimateCertainty(newStateVars);
//				final double familiarity3 = estimateWeightCertainty(modeler.getTransitionsModule().getNeuralNetwork().getInputNodes(), false);
//				System.out.println(Utils.stringArray(actionVars, 1));
				final double familiarity4 = estimateWeightCertainty(modeler.getTransitionsModule().getNeuralNetwork().getInputNodes(), true);
//				final double familiarity5 = maxWeight(newStateVars);
				if (jointAdjustments > 0 && modeler.getFamiliarityModule() != null)
					newStateVars = modeler.upJointOutput(allVars, allVars.length - newStateVars.length, jointAdjustments);
				//				final double stateSimilarity = i == 0 ? stateEquality(stateVars, newStateVars) : 0;
				// TODO look at weights from action inputs!
//				System.out.println(actionVars[0] + "	" + actionVars[1] + "	" + actionVars[2] + "	"
//						+ actionVars[3] + "	" + familiarity1 + "	" + familiarity2
//						+ "	" + familiarity3 + "	" + familiarity4);// + Utils.stringArray(newStateVars, 4));
				if (mra != null) mra.observeState(i, 1, familiarity4);//familiarity5 * familiarity4 * (stateSimilarity + 1));
				stateVars = probabilitySkewing(newStateVars, skewFactor);
			}
		}
		modeler.clearWorkingMemory();
	}
	
	public static double estimateWeightCertainty(Collection<? extends Node> nodes, boolean useCum) {
		double sum = 0;
		double denom = 0;
		String s = "";
		for (Node n : nodes) {
			final double a = n.getActivation();
			if (a <= 0) continue;
			denom += a;
			for (Connection c : n.getOutputConnections()) {
				s += c.toString() + "	" + Utils.round(c.getWeight().getCumSqrChg() * a,4) + "	";
				if (useCum) {
					sum += c.getWeight().getCumSqrChg() * a;
				} else {
					double e = c.getWeight().getWeight();
					sum += e*e * a;
				}
			}
		}
//		System.out.println(s);
		return sum / denom;
	}
	public static double maxWeight(double[] in) {
		double max = 0;
		for (double d : in) if (d > max) max = d;
		return max;
	}
	public static double estimateCertainty(double[] in) {
		double sum = 0;
		for (double d : in) {
			double e = d - 0.5;
			sum += e*e;
		}
		return sum / in.length;
	}
	public static double stateSimilarity(double[] s1, double[] s2) {
		if (s1.length != s2.length) throw new IllegalStateException("state sizes must be equal");
		double sumdiff = 0;
		for (int i = 0; i < s1.length; i++) {
			double diff = s1[i] - s2[i];
			sumdiff += diff*diff;
		}
		return 1 - Math.sqrt(sumdiff / s1.length);
	}
	public static double stateEquality(double[] s1, double[] s2) {
		if (s1.length != s2.length) throw new IllegalStateException("state sizes must be equal");
		double sumsame = 0;
		for (int i = 0; i < s1.length; i++) {
			double diff = s1[i] - s2[i];
			if (diff*diff < 0.25) sumsame ++;
		}
		return sumsame == s1.length ? 1 : 0;
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
