package reasoner;

import java.util.*;

import ann.*;
import modeler.*;

public class NeuralGibbs {

	private static double[] initializeGuess(ModelLearner modeler, double[] currState, double[] action, boolean ff) {
		Collection<? extends Node> outputsVTA = modeler.getTransitionsModule().getNeuralNetwork().getOutputNodes();
		double[] newStateVars = new double[currState.length];
		if (ff) {
			modeler.observeAction(action);
			modeler.observePreState(currState);
			modeler.feedForward();
		}
		int j = 0;
		for (Node n : outputsVTA) newStateVars[j++] = n.getActivation();
		newStateVars = Foresight.probabilisticRoundingUnnormalized(newStateVars);
		return newStateVars;
	}
	
	public static Collection<double[]> guess(ModelLearner modeler, double[] currState, double[] action,
			int numSamples) {
		Collection<double[]> result = new ArrayList<double[]>();
		while(result.size() < numSamples) {
			result.add(initializeGuess(modeler, currState, action, true));
		}
		return result;
	}
	
	public static Collection<double[]> sample(ModelLearner modeler, double[] currState, double[] action,
			int numSamples, int burnIn, int thinning) {
		Collection<? extends Node> inputsJDM = modeler.getFamiliarityModule().getNeuralNetwork().getInputNodes();
		ArrayList<? extends Node> outputsJDM = modeler.getFamiliarityModule().getNeuralNetwork().getOutputNodes();

		Collection<double[]> result = new ArrayList<double[]>();
		List<Integer> ord = new ArrayList<Integer>();
		int postStateIndex = currState.length + action.length;
		for (int i = 0; i < currState.length; i++) ord.add(i);
		boolean firstTime = true;
		while(result.size() < numSamples) {
			double[] newStateVars = initializeGuess(modeler, currState, action, firstTime);
			firstTime = false;
			Collections.shuffle(ord);
			for (int s = 0; s < burnIn; s++) {
				for (int i : ord) {
					double[] ins = ModelLearnerHeavy.concatVars(currState, action, newStateVars);
//					ins[postStateIndex + i] = Math.random(); // set self evidence to uncertain
					FFNeuralNetwork.feedForward(inputsJDM, ins);
//					System.out.println("P(y"+i+"|"+Utils.stringArray(ins, 1) + " = " + outputsJDM.get(i).getActivation());
					newStateVars[i] = Math.random() < outputsJDM.get(i).getActivation() ? 1 : 0;
				}
			}
			double[] newSample = new double[newStateVars.length];
			System.arraycopy(newStateVars, 0, newSample, 0, newStateVars.length);
//			System.out.println(Utils.stringArray(newStateVars, 2));
//			System.out.println("----------");
			result.add(newSample);
		}
		return result;
	}
}
