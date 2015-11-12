package reasoner;

import java.util.*;

import ann.Utils;
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
	public static Planner createMonteCarloPlanner(final ModelLearnerHeavy modeler, final int numSteps,
			final int numRuns, final RewardFunction rewardFn, final boolean holdAction,
			final double discountRate, final int jointAdjst) {
		return createMonteCarloPlanner(modeler, numSteps, numRuns, rewardFn, holdAction, discountRate,
				jointAdjst, null, null);
	}
	/** must use sigmoidal if representing probabilities 
	 * @param actionTranslator 
	 * @param stateTranslator */
	public static Planner createMonteCarloPlanner(final ModelLearner modeler, final int numSteps,
			final int numRuns, final RewardFunction rewardFn, final boolean holdAction,
			final double discountRate, final int jointAdjs,
			final EnvTranslator stateTranslator, final EnvTranslator actionTranslator) {
		return new Planner() {
			@Override
			public double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
					double explorePct, double rewardMutationRate) {
				if (Math.random() < explorePct) return RandomUtils.randomOf(actionChoices);
				double[] bestActionChoice = null;
				double bestReward = Double.NEGATIVE_INFINITY;
				boolean DEBUG = false;
//				if (stateTranslator == null || actionTranslator == null) DEBUG = false;
				String s = "";
				if (DEBUG && stateTranslator != null) s = debug1(s + " I :", stateTranslator.fromNN(initialStateVars));
				for (double[] actionChoice : actionChoices) {
					// TODO include reward function in montecarlo so it can compute at each step
					final MultiRewardAssessment mra = new MultiRewardAssessment(rewardFn, discountRate);
					double[] outputs = Foresight.montecarlo(modeler, initialStateVars, actionChoice,
							holdAction ? null : actionChoices, mra, numSteps, numRuns, jointAdjs, 0);
					if (DEBUG && actionTranslator != null) s = debug1(s + " A :", actionTranslator.fromNN(actionChoice));
					if (DEBUG) s = debug1(s + " O :", outputs);
//					double reward = rewardFn.getReward(outputs);
					double reward = mra.getExpReward();
					reward = mutateReward(reward, rewardMutationRate);
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
	
	public static double mutateReward(double reward, double mutRate) {
		return reward * Math.exp(2 * Math.random() * mutRate - mutRate) + (Math.random() - 0.5)/100000;
	}
	
	private static final RewardFunction EXPLORE_UNFAMILIARITY_REWARD = new RewardFunction() {
		@Override
		public double getReward(double[] familiarity) {
			return -familiarity[0];
		}
	};
	
	public static Planner createKWIKExplorer(final ModelLearnerHeavy modeler, final int numSteps, final int numRuns,
			final EnvTranslator stateTranslator, final EnvTranslator actionTranslator) {
		return new Planner() {
			@Override
			public double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
					double explorePct, double rewardMutationRate) {
				List<double[]> remainingActionChoices = new ArrayList<double[]>();
				remainingActionChoices.addAll(actionChoices);
				for (double[] action : actionChoices) if (Foresight.knowsWhatItKnows(modeler, initialStateVars,
						action, 10, .22)) remainingActionChoices.remove(action);
				return RandomUtils.randomOf(remainingActionChoices.isEmpty() ? actionChoices : remainingActionChoices);
			}
		};
	}
	
	public static Planner createNoveltyExplorer(final ModelLearnerHeavy modeler, final int numSteps, final int numRuns,
			final EnvTranslator stateTranslator, final EnvTranslator actionTranslator) {
		return new Planner() {
			@Override
			public double[] getOptimalAction(double[] initialStateVars, List<double[]> actionChoices,
					double explorePct, double rewardMutationRate) {
				if (Math.random() < explorePct) return RandomUtils.randomOf(actionChoices);
				double[] bestActionChoice = null;
				double bestReward = Double.NEGATIVE_INFINITY;
				boolean DEBUG = true;
				String s = "";
//				if (DEBUG) s = debug1(s + " I :", stateTranslator == null ? initialStateVars : stateTranslator.fromNN(initialStateVars));
				for (double[] actionChoice : actionChoices) {
					final MultiRewardAssessment mra = new MultiRewardAssessment(EXPLORE_UNFAMILIARITY_REWARD, 0.05);
					Foresight.terraIncognita(modeler, initialStateVars, actionChoice, actionChoices,
							mra, numSteps, numRuns, 10, 0);
					if (DEBUG) s = debug1(s + " A :", actionTranslator == null ? actionChoice : actionTranslator.fromNN(actionChoice));
					double reward = mra.getExpReward();
					reward *= Math.exp(2 * Math.random() * rewardMutationRate - rewardMutationRate);
					if (DEBUG) s += "R=	" + Utils.round(reward,4) + "	";
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
