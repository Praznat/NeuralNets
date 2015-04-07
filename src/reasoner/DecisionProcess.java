package reasoner;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import modeler.*;
import deepnets.Node;

public class DecisionProcess {
	
	private final ModelLearner modeler;
	private final List<double[]> actionChoices;
	private final int depth;
	private final int numRuns;
	private final int jointAdjustments;
	private final double skewFactor;
	private final double discountRate;
	private final double confusionRate;
	
	private boolean isLogging;
	
	private Map<DiscreteState, ActionRewardForecast> memo = new HashMap<DiscreteState, ActionRewardForecast>();
	
	public DecisionProcess(ModelLearner modeler, List<double[]> actionChoices, int depth, int numRuns,
			int jointAdjustments, double skewFactor, double discountRate, double confusionRate) {
		this.modeler = modeler;
		this.actionChoices = actionChoices;
		this.depth = depth;
		this.numRuns = numRuns;
		this.jointAdjustments = jointAdjustments;
		this.skewFactor = skewFactor;
		this.discountRate = discountRate;
		this.confusionRate = confusionRate;
	}
	
	public Map<DiscreteState,AtomicInteger> getImmediateStateGraphForAction(double[] currState, double[] action) {
		Map<DiscreteState,AtomicInteger> result = new HashMap<DiscreteState,AtomicInteger>();
		Collection<? extends Node> outputsVTA = modeler.getTransitionsModule().getNeuralNetwork().getOutputNodes();
		
		modeler.observeAction(action);
		modeler.observePreState(currState);
		modeler.feedForward();

		double[] newStateVars = new double[currState.length];
		int j = 0;
		for (Node n : outputsVTA) newStateVars[j++] = n.getActivation(); // TODO move this out of for loop
		for (int r = 0; r < numRuns; r++) {
			double[] v = Foresight.probabilisticRoundingUnnormalized(
					Foresight.probabilitySkewing(newStateVars, skewFactor)); // adds noise
			double[] allVars = ModelLearnerHeavy.concatVars(currState, action, v);
			if (jointAdjustments > 0 && modeler.getFamiliarityModule() != null)
				v = modeler.upJointOutput(allVars, allVars.length - newStateVars.length, jointAdjustments);
			DiscreteState ds = new DiscreteState(v);
//			if (ds.toString().equals("")) {
//				System.out.println("shit");
//			}
			AtomicInteger count = result.get(ds);
			if (count == null) result.put(ds, count = new AtomicInteger(1));
			else count.incrementAndGet();
		}
		return result;
	}
	
	private double evaluate(Forecast forecast, RewardFunction rewardFn, int depthLeft) {
		Iterator<Map.Entry<DiscreteState,Double>> iter = forecast.entrySet().iterator();
		double wgtSum = 0;
	    while (iter.hasNext()) {
	        Map.Entry<DiscreteState,Double> entry = iter.next();
	        double[] state = entry.getKey().rawState;
	        double reward = rewardFn.getReward(state);
	        if (depthLeft > 0) reward += getBestARFFromState(state, rewardFn, depthLeft-1).getReward();
	        double probability = entry.getValue();
	        wgtSum += reward * probability;
	    }
		return wgtSum;
	}
	
	public ActionRewardForecast getBestARFFromState(double[] state,
			RewardFunction rewardFn, int remDepth) {
		DiscreteState ds = new DiscreteState(state);
//		ActionRewardForecast stored = memo.get(ds);
//		if (stored != null) return stored;
		double[] bestAction = null;
		Forecast bestForecast = null;
		double bestReward = Double.NEGATIVE_INFINITY;
		for (double[] action : actionChoices) {
			Forecast forecast = new Forecast(getImmediateStateGraphForAction(state, action));
			double reward = evaluate(forecast, rewardFn, remDepth);
			if (isLogging) log(ds, action, forecast, reward, remDepth);
			if (remDepth == this.depth) reward = Planner.mutateReward(reward, confusionRate);
			if (reward > bestReward) {
				bestReward = reward;
				bestAction = action;
				bestForecast = forecast;
			}
		}
		ActionRewardForecast arf = new ActionRewardForecast(bestAction, bestReward * (1-discountRate), bestForecast);
		memo.put(ds, arf);
		return arf;
	}
	
	private void log(DiscreteState fromState, double[] action, Forecast forecast, double reward, int remDepth) {
		StringBuilder sb = new StringBuilder();
		for (int d = remDepth; d < this.depth; d++) sb.append("	");
		sb.append("S: ");
		sb.append(fromState);
		sb.append("	A: ");
		for (double d : action) sb.append(d+",");
		sb.append("	F: ");
		sb.append(forecast);
		sb.append("	r: ");
		sb.append(reward);
		System.out.println(sb.toString());
	}
	
	public double[] buildDecisionTree(double[] state, RewardFunction rewardFn, int depth,
			double discRate, boolean fromScratch) {
		if (fromScratch) memo.clear();
		ActionRewardForecast arf = getBestARFFromState(state, rewardFn, depth);
		return arf.getAction();
	}
	
	public void setLogging(boolean l) {
		isLogging = l;
	}
	
	@Override
	public String toString() {
		return memo.toString();
	}
}
