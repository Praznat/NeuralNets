package transfer;

import java.util.ArrayList;
import java.util.Collection;

import ann.Utils;
import modeler.ModelLearner;
import modeler.TransitionMemory;
import reasoner.Foresight;
import reasoner.MultiRewardAssessment;
import reasoner.RewardFunction;

public class ModelSelector {

	private static final int DEFAULT_RUNS = 5;
	private static final int DEFAULT_JOINTS = 1;
	
	private Collection<SimulationScore> storedScores = new ArrayList<SimulationScore>();

	public ModelSelector() {}
	
	public ModelSelector(String... names) {
		loadFromFiles(names);
	}
	
	public void loadFromFiles(String... names) {
		for (String name : names) {
			storedScores.add(new SimulationScore(name));
		}
	}
	
	public void loadModeler(String name, ModelLearner modeler) {
		storedScores.add(new SimulationScore(name, modeler));
	}
	
	public SimulationScore observeWorkingModelTransitions(ModelLearner workingModeler, int numTransitions) {
		return observeWorkingModelTransitions(workingModeler, numTransitions, DEFAULT_RUNS);
	}
	/**
	 * returns best model learner after observing transitions
	 */
	public SimulationScore observeWorkingModelTransitions(ModelLearner workingModeler, int numTransitions, int runs) {
		Collection<TransitionMemory> transitions = workingModeler.getExperience().getBatch(numTransitions, true);
		double bestScore = 0;
		SimulationScore best = null;
		for (SimulationScore storedScore : storedScores) {
			System.out.println(storedScore);
			double score = 0;
			for (TransitionMemory tm : transitions) {
				score += calcScoreFromTransition(storedScore.getModeler(), tm, runs, DEFAULT_JOINTS);
			}
			storedScore.observeNewScore(score);
			if (score > bestScore) {
				bestScore = score;
				best = storedScore;
			}
		}
		return best;
	}
	
	/**
	 * given a transition memory, rates the given modeler's ability to predict the same post-state
	 * from the same pre-state and action
	 */
	private static double calcScoreFromTransition(ModelLearner modeler, TransitionMemory tm, int numRuns, int joints) {
		final double[] targetPostState = tm.getPostState();
		MultiRewardAssessment mra = new MultiRewardAssessment(new RewardFunction() {
			@Override
			public double getReward(double[] stateVars) {
				double [] predictedPostState = Foresight.probabilisticRoundingUnnormalized(stateVars);
				return Foresight.stateSimilarity(predictedPostState, targetPostState);
//				return Foresight.stateEquality(predictedPostState, targetPostState);
			}
		}, 0);
		Foresight.montecarlo(modeler, tm.getPreState(), tm.getAction(), null, mra, 1, numRuns, joints, 0);
		final double pReachTarget = mra.getExpReward();
		System.out.println(pReachTarget);
		return pReachTarget;
	}
}
