package modeler;

import deepnets.ExperienceReplay;

public abstract class ModelLearner {

	protected double[] workingPreState, workingAction, workingPostState;
	protected final ExperienceReplay<TransitionMemory> experienceReplay;
	
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
	
	public void saveMemory() {
		if (experienceReplay == null) throw new IllegalStateException("must define experience replay size");
		experienceReplay.addMemory(createMemory());
		workingPreState = null;
		workingAction = null;
		workingPostState = null; // to create exception if you try to save memory without reobserving
	}
	
	// TODO learn online
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

	public abstract ModelerModule getTransitionsModule();

//	public abstract double getFamiliarity(double[] allVars);
//	public abstract double[] upFamiliarity(double[] allVars, int postLen, int jointAdjustments, int epochs,
//			double lRate, double mRate, double sRate, double maxShift);
//	public abstract double[] upFamiliarity(TransitionMemory tm, int jointRounds,
//			int epochs, double lRate, double mRate, double sRate, double maxShift);


}
