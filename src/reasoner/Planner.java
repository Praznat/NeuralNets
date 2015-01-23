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
	public static Planner createMonteCarloPlanner(final VariableTransitionApproximator modeler, final int numSteps, final int numRuns,
			final RewardFunction rewardFn, final EnvTranslator stateTranslator, final EnvTranslator actionTranslator) {
		return new Planner() {
			@Override
			public double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
					double explorePct, double rewardMutationRate) {
				if (Math.random() < explorePct) return RandomUtils.randomOf(actionChoices);
				double[] bestActionChoice = null;
				double bestReward = Double.NEGATIVE_INFINITY;
				String s = "";
				s = debug1(s + " I :", stateTranslator.fromNN(initialStateVars));
				for (double[] actionChoice : actionChoices) {
					double[] outputs = Foresight.montecarlo(modeler, initialStateVars, actionChoice,
							numSteps, numRuns, 0.00);
					s = debug1(s + " A :", actionTranslator.fromNN(actionChoice));
					s = debug1(s + " O :", outputs);
					double reward = rewardFn.getReward(outputs);
					reward *= Math.exp(2 * Math.random() * rewardMutationRate - rewardMutationRate);
					s += "R=	" + reward + "	";
					if (reward > bestReward) {
						bestReward = reward;
						bestActionChoice = actionChoice;
					}
				}
				System.out.println(s);
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
