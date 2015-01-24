package reasoner;

import java.util.List;

import modeler.*;
import utils.RandomUtils;

public abstract class Planner {

	public abstract double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
			double explorePct, double rewardMutationRate);
	
	/** must use sigmoidal if representing probabilities 
	 * @param actionTranslator 
	 * @param stateTranslator */
	public static Planner createMonteCarloPlanner(final ModelLearner modeler, final int numSteps, final int numRuns,
			final RewardFunction rewardFn, final EnvTranslator stateTranslator, final EnvTranslator actionTranslator) {
		return new Planner() {
			@Override
			public double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
					double explorePct, double rewardMutationRate) {
				if (Math.random() < explorePct) return RandomUtils.randomOf(actionChoices);
				double[] bestActionChoice = null;
				double bestReward = Double.NEGATIVE_INFINITY;
				final boolean DEBUG = false;
				String s = "";
				if (DEBUG) s = debug1(s + " I :", stateTranslator.fromNN(initialStateVars));
				for (double[] actionChoice : actionChoices) {
					double[] outputs = Foresight.montecarlo(modeler, initialStateVars, actionChoice,
							numSteps, numRuns, 0.00);
					if (DEBUG) s = debug1(s + " A :", actionTranslator.fromNN(actionChoice));
					if (DEBUG) s = debug1(s + " O :", outputs);
					double reward = rewardFn.getReward(outputs);
					reward *= Math.exp(2 * Math.random() * rewardMutationRate - rewardMutationRate);
					if (DEBUG) s += "R=	" + reward + "	";
					if (reward > bestReward) {
						bestReward = reward;
						bestActionChoice = actionChoice;
					}
				}
				if (DEBUG) System.out.println(s);
				return bestActionChoice;
			}
		};
	}
	
	private static String debug1(String pre, double[] envVars) {
		for (double a : envVars) pre += (Math.round(a*10000)/10000.0) + "	";
		pre += "	";
		return pre;
	}
	
}
