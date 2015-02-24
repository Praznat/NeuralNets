package reasoner;

import java.util.List;

import modeler.*;
import utils.RandomUtils;

public abstract class Planner {

	public abstract double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
			double explorePct, double rewardMutationRate);
	
	public static Planner createRandomChimp() {
		return new Planner() {
			@Override
			public double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
					double explorePct, double rewardMutationRate) {
				return RandomUtils.randomOf(actionChoices);
			}
		};
	}
	public static Planner createMonteCarloPlanner(final ModelLearnerHeavy modeler, final int numSteps, final int numRuns,
			final RewardFunction rewardFn) {
		return createMonteCarloPlanner(modeler, numSteps, numRuns, rewardFn, null, null);
	}
	/** must use sigmoidal if representing probabilities 
	 * @param actionTranslator 
	 * @param stateTranslator */
	public static Planner createMonteCarloPlanner(final ModelLearnerHeavy modeler, final int numSteps, final int numRuns,
			final RewardFunction rewardFn, final EnvTranslator stateTranslator, final EnvTranslator actionTranslator) {
		return new Planner() {
			@Override
			public double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
					double explorePct, double rewardMutationRate) {
				if (Math.random() < explorePct) return RandomUtils.randomOf(actionChoices);
				double[] bestActionChoice = null;
				double bestReward = Double.NEGATIVE_INFINITY;
				boolean DEBUG = false;
				if (stateTranslator == null || actionTranslator == null) DEBUG = false;
				String s = "";
				if (DEBUG) s = debug1(s + " I :", stateTranslator.fromNN(initialStateVars));
				for (double[] actionChoice : actionChoices) {
					// TODO include reward function in montecarlo so it can compute at each step
					final MultiRewardAssessment mra = new MultiRewardAssessment(rewardFn, 0);
					double[] outputs = Foresight.montecarlo(modeler, initialStateVars, actionChoice,
							mra, numSteps, numRuns, 10, 0.1);
					if (DEBUG) s = debug1(s + " A :", actionTranslator.fromNN(actionChoice));
					if (DEBUG) s = debug1(s + " O :", outputs);
//					double reward = rewardFn.getReward(outputs);
					double reward = mra.getExpReward();
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
