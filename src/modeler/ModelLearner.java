package modeler;

import java.util.*;

import ann.*;
import reasoner.DiscreteState;

public abstract class ModelLearner {

	protected double[] workingPreState, workingAction, workingPostState;
	protected final ExperienceReplay<TransitionMemory> experienceReplay;
	
	protected boolean isRecordingTraining;
	protected ArrayList<Double> trainingErrorLog = new ArrayList<Double>();
	
	public ModelLearner(int expReplaySize) {
		this.experienceReplay = expReplaySize > 0 ? new ExperienceReplay<TransitionMemory>(expReplaySize) : null;
	}
	
	public void observePreState(double... values) {
		workingPreState = values;
	}
	public void observeAction(double... values) {
		workingAction = values;
	}
	public void observePostState(double... values) {
		workingPostState = values;
	}
	
	protected TransitionMemory createMemory() {
		return new TransitionMemory(workingPreState, workingAction, workingPostState);
	}
	
	public TransitionMemory saveMemory() {
		if (experienceReplay == null) throw new IllegalStateException("must define experience replay size");
		TransitionMemory tm = createMemory();
		saveMemory(tm);
		clearWorkingMemory();
		return tm;
	}
	public void saveMemory(TransitionMemory tm) {
		experienceReplay.addMemory(tm);
	}
	public void saveMemories(Collection<TransitionMemory> memories) {
		if (experienceReplay == null) throw new IllegalStateException("must define experience replay size");
		for (TransitionMemory tm : memories) experienceReplay.addMemory(tm);
	}
	
	public void clearWorkingMemory() {
		workingPreState = null;
		workingAction = null;
		workingPostState = null; // to create exception if you try to save memory without reobserving
	}
	
	public void clearExperience() {
		experienceReplay.clear();
	}
	
	// TODO learn online
	public abstract void learnOnline(double lRate, double mRate, double sRate);
	
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations) {
		learnFromMemory(lRate, mRate, sRate, resample, iterations, 0, 0);
	}
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, double stopAtErrThreshold) {
		learnFromMemory(lRate, mRate, sRate, resample, iterations, 0, stopAtErrThreshold);
	}
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, long displayProgressMs) {
		learnFromMemory(lRate, mRate, sRate, resample, iterations, displayProgressMs, 0);
	}
	protected static long debugTime(String s, long ms) {
		long nowMs = System.currentTimeMillis();
		System.out.println(s + (nowMs - ms));
		return nowMs;
	}
	public abstract void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, long displayProgressMs, double stopAtErrThreshold);
	
	public abstract void feedForward();
	public double[] upJointOutput(TransitionMemory tm, int rounds) {
		return upJointOutput(tm.getAllVars(), tm.getPreStateAndAction().length, rounds);
	}
	public abstract double[] upJointOutput(double[] vars, int postStateIndex, int rounds);

	public abstract ModelNeuralNet getTransitionsModule();
	public abstract ModelNeuralNet getFamiliarityModule();
	
	public void recordTraining() {
		this.isRecordingTraining = true;
	}
	
	public ArrayList<Double> getTrainingLog() {
		return trainingErrorLog;
	}

	public double getPctMastered() {
		return getPctMastered(-1, Integer.MAX_VALUE);
	}
	public double getPctMastered(int minVar, int maxVar) {
		double sum = 0;
		ModelNeuralNet vta = getTransitionsModule();
		allmemo:
		for (TransitionMemory tm : experienceReplay.getBatch()) {
			observeAction(tm.action);
			observePreState(tm.preStateVars);
			feedForward();
			int i = 0;
			if (tm.postStateVars.length < maxVar) maxVar = tm.postStateVars.length;
			for (Node n : vta.getNeuralNetwork().getOutputNodes()) {
				if (i >= minVar) {
					if (i >= maxVar) break;
					if (Math.round(n.getActivation()) != Math.round(tm.postStateVars[i])) {
						continue allmemo; // if wrong activation, dont increment sum
					}
				}
				i++;
			}
			sum++;
//			System.out.println(new DiscreteState(tm.postStateVars));
		}
		return sum / experienceReplay.getSize();
	}
	
	public void filterExperienceToBooleans() {
		Set<DiscreteState> uniqueMemories = new HashSet<DiscreteState>();
		for (TransitionMemory tm : experienceReplay.getBatch()) {
			uniqueMemories.add(new DiscreteState(tm.getAllVars()));
		}
		experienceReplay.clear();
		int sa = workingPreState.length + workingAction.length;
		for (DiscreteState ds : uniqueMemories) {
			double[] rs = ds.getRawState();
			TransitionMemory tm = new TransitionMemory(Arrays.copyOfRange(rs, 0, sa),
					workingPreState.length, Arrays.copyOfRange(rs, sa, rs.length));
			experienceReplay.addMemory(tm);
		}
	}
	
	public ExperienceReplay<TransitionMemory> getExperience() {
		return experienceReplay;
	}

	public double[] newStateVars(double[] stateVars, double[] action, int jointAdjustments) {
		observeAction(action);
		observePreState(stateVars);
		feedForward();
		
		Collection<? extends Node> outputs = getTransitionsModule().getNeuralNetwork().getOutputNodes();
		double[] newStateVars = new double[stateVars.length];
		int j = 0;
		for (Node n : outputs) newStateVars[j++] = n.getActivation();
		double[] allVars = ModelLearnerHeavy.concatVars(stateVars, action, newStateVars);
		if (jointAdjustments > 0 && getFamiliarityModule() != null)
			newStateVars = upJointOutput(allVars, allVars.length - newStateVars.length, jointAdjustments);
		return newStateVars;
	}

//	public abstract double getFamiliarity(double[] allVars);
//	public abstract double[] upFamiliarity(double[] allVars, int postLen, int jointAdjustments, int epochs,
//			double lRate, double mRate, double sRate, double maxShift);
//	public abstract double[] upFamiliarity(TransitionMemory tm, int jointRounds,
//			int epochs, double lRate, double mRate, double sRate, double maxShift);


}
