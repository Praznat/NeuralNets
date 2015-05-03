package reasoner;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import modeler.*;
import deepnets.*;

public class DecisionProcess {
	
	public static enum LogLevel {NONE, LO, MID, HI};
	
	private final ModelLearner modeler;
	private final List<double[]> actionChoices;
	private final int depth;
	private final int numRuns;
	private final int jointAdjustments;
	private final double skewFactor;
	private final double discountRate;
	private final double cutoffProb;
	
	private LogLevel logLevel = LogLevel.NONE;
	
	private Map<DiscreteState, ActionRewardForecast> memo = new HashMap<DiscreteState, ActionRewardForecast>();
	
	public DecisionProcess(ModelLearner modeler, List<double[]> actionChoices, int depth, int numRuns,
			int jointAdjustments, double skewFactor, double discountRate) {
		this(modeler, actionChoices, depth, numRuns, jointAdjustments, skewFactor, discountRate, 0);
	}
	public DecisionProcess(ModelLearner modeler, List<double[]> actionChoices, int depth, int numRuns,
			int jointAdjustments, double skewFactor, double discountRate, double cutoffProb) {
		this.modeler = modeler;
		this.actionChoices = actionChoices;
		this.depth = depth;
		this.numRuns = numRuns;
		this.jointAdjustments = jointAdjustments;
		this.skewFactor = skewFactor;
		this.discountRate = discountRate;
		this.cutoffProb = cutoffProb;
	}
	public Map<DiscreteState,AtomicInteger> getImmediateStateGraphForActionGibbs(double[] currState, double[] action) {
		Map<DiscreteState,AtomicInteger> result = new HashMap<DiscreteState,AtomicInteger>();
		int burnIn = 10; // unit test ProbabilityTracking if you change this!
		Collection<double[]> samples = jointAdjustments > 0
				? NeuralGibbs.sample(modeler, currState, action, numRuns, burnIn, -666)
				: NeuralGibbs.guess(modeler, currState, action, numRuns);
		for (double[] sample : samples) {
			DiscreteState ds = new DiscreteState(sample);
			AtomicInteger count = result.get(ds);
			if (count == null) result.put(ds, count = new AtomicInteger(1));
			else count.incrementAndGet();
		}
		return result;
	}
	
	@Deprecated
	public Map<DiscreteState,AtomicInteger> getImmediateStateGraphForActionOLD(double[] currState, double[] action) {
		Map<DiscreteState,AtomicInteger> result = new HashMap<DiscreteState,AtomicInteger>();
		Collection<? extends Node> outputsVTA = modeler.getTransitionsModule().getNeuralNetwork().getOutputNodes();
		
		modeler.observeAction(action);
		modeler.observePreState(currState);
		modeler.feedForward();

		double[] newStateVars = new double[currState.length];
		int j = 0;
		for (Node n : outputsVTA) newStateVars[j++] = n.getActivation();
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
	
	private double evaluate(Forecast forecast, RewardFunction rewardFn, int depthLeft,
			double confusion, DiscreteState thisState) {
		Iterator<Map.Entry<DiscreteState,Double>> iter = forecast.entrySet().iterator();
		double wgtSum = 0;
	    while (iter.hasNext()) {
	        Map.Entry<DiscreteState,Double> entry = iter.next();
	        double[] state = entry.getKey().rawState;
	        double reward = rewardFn.getReward(state);
	        boolean sameState = Utils.sameArray(state, thisState.getRawState());
	        if (depthLeft > 0 && !sameState)
	        	reward += getBestARFFromState(state, rewardFn, depthLeft-1, confusion).getReward();
	        double probability = entry.getValue();
	        wgtSum += reward * probability;
	    }
		return wgtSum;
	}
	
	public ActionRewardForecast getBestARFFromState(double[] state,
			RewardFunction rewardFn, int remDepth, double confusion) {
		DiscreteState ds = new DiscreteState(state);
//		ActionRewardForecast stored = memo.get(ds);
//		if (stored != null) return stored;
		double[] bestAction = null;
		Forecast bestForecast = null;
		double bestReward = Double.NEGATIVE_INFINITY;
		for (double[] action : actionChoices) {
//			Forecast forecast = new Forecast(getImmediateStateGraphForActionGibbs(state, action), cutoffProb);
			Forecast forecast = new Forecast(getImmediateStateGraphForActionOLD(state, action), cutoffProb);
			double reward = evaluate(forecast, rewardFn, remDepth, confusion, ds);
			log(logLevel, ds, action, forecast, reward, remDepth);
			if (remDepth == this.depth) reward = Planner.mutateReward(reward, confusion);
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
	
	private void log(LogLevel logLevel, DiscreteState fromState, double[] action, Forecast forecast, double reward, int remDepth) {
		if (logLevel == LogLevel.NONE) return;
		if (logLevel == LogLevel.LO) {
			System.out.println(action + "	" + remDepth);
			return;
		}
		if (logLevel == LogLevel.MID && reward == 0) return;
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
			double discRate, boolean fromScratch, double confusion) {
		if (fromScratch) memo.clear();
		ActionRewardForecast arf = getBestARFFromState(state, rewardFn, depth, confusion);
		// TODO compare this forecast against actual resulting state
		// if resulting state is not in forecast, add previous state to UNLEARNT set
		// incorporate UNLEARNT states into decision process somehow
		// occasionally learn appropriate transitions from UNLEARNT states
		return arf.getAction();
	}
	
	public void setLogging(LogLevel l) {
		logLevel = l;
	}
	
	@Override
	public String toString() {
		return memo.toString();
	}
}
