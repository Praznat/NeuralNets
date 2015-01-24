package modeler;

import java.util.*;

import utils.RandomUtils;
import deepnets.*;

public class ModelLearner {

	private final ExperienceReplay<TransitionMemory> experienceReplay;
	private final TransitionRealismAssessor modelTRA;
	private final VariableTransitionApproximator modelVTA;
	
	private double[] workingPreState, workingAction, workingPostState;
	
	public ModelLearner(int errorHalfLife, int[] numHiddenVTA, int[] numHiddenTRA,
			ActivationFunction actFn, int expReplaySize) {
		this.experienceReplay = expReplaySize > 0 ? new ExperienceReplay<TransitionMemory>(expReplaySize) : null;
		modelVTA = new VariableTransitionApproximator(actFn, numHiddenVTA, errorHalfLife);
		modelTRA = new TransitionRealismAssessor(actFn, numHiddenTRA, errorHalfLife);
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
	
	private TransitionMemory createMemory() {
		return new TransitionMemory(workingPreState, workingAction, workingPostState);
	}
	
	public void saveMemory() {
		if (experienceReplay == null) throw new IllegalStateException("must define experience replay size");
		experienceReplay.addMemory(createMemory());
		workingPreState = null;
		workingAction = null;
		workingPostState = null; // to create exception if you try to save memory without reobserving
	}
	
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations) {
		learnFromMemory(lRate, mRate, sRate, resample, iterations, 0, 0);
	}
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, double stopAtErrThreshold) {
		learnFromMemory(lRate, mRate, sRate, resample, iterations, 0, stopAtErrThreshold);
	}
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, int batchSize) {
		learnFromMemory(lRate, mRate, sRate, resample, iterations, batchSize, 0);
	}
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, int batchSize, double stopAtErrThreshold) {
		for (int i = 0; i < iterations; i++) {
			Collection<TransitionMemory> memories = batchSize > 0 ? experienceReplay.getBatch(batchSize, resample)
					: experienceReplay.getBatch(resample);
			for (TransitionMemory m : memories) {
				getModelVTA().analyzeTransition(m, lRate, mRate, sRate);
				getModelTRA().analyzeTransition(m, lRate, mRate, sRate);
			}
			if (stopAtErrThreshold > 0 && getModelVTA().getConfidenceEstimate() < stopAtErrThreshold) return; // TODO TRA?
		}
	}
	
	public void feedForward() {
		TransitionMemory tm = createMemory();
		FFNeuralNetwork.feedForward(getModelVTA().getNeuralNetwork().getInputNodes(), tm.getPreStateAndAction());
		if (workingPostState != null) FFNeuralNetwork.feedForward(getModelTRA().getNeuralNetwork().getInputNodes(), tm.getAllVars());
	}
	
	public void testit(int times, double[] mins, double[] maxes,
			EnvTranslator stateTranslator, EnvTranslator actTranslator, List<double[]> actions) {
		testit(times, mins, maxes, stateTranslator, actTranslator, actions, false);
	}
	public void testit(int times, double[] mins, double[] maxes,
			EnvTranslator stateTranslator, EnvTranslator actTranslator, List<double[]> actions, boolean useRaw) {
		if (mins.length != maxes.length) throw new IllegalStateException("mins must equal maxes");
		double[] outputActivation = null;
		for (int t = 0; t < times; t++) {
			final double[] state = new double[mins.length];
			for (int i = 0; i < state.length; i++) state[i] = RandomUtils.randBetween(mins[i], maxes[i]);
			double[] inNN = stateTranslator.toNN(state);
			observePreState(inNN);
			if (outputActivation == null) outputActivation = new double[inNN.length];
			String s = "";
			for (double d : (useRaw ? inNN : state)) s += r(d) + "	";
			for (double[] action : actions) {
				s += "|	";
				observeAction(action);
				feedForward();
				int i = 0;
				for (Node n : getModelVTA().getNeuralNetwork().getOutputNodes()) outputActivation[i++] = n.getActivation();
				if (useRaw) {
					for (double d : action) s += r(d) + "	";
					s += ":	";
					for (double d : outputActivation) s += r(d) + "	";
				} else {
					double[] acty = actTranslator.fromNN(action);
					for (double d : acty) s += r(d) + "	";
					s += ":	";
					double[] outy = stateTranslator.fromNN(outputActivation);
					for (double d : outy) s += r(d) + "	";
				}
			}
			System.out.println(s);
		}
	}
	
	protected static String r(double activation) {
		return String.valueOf(((int)Math.round(activation*1000))/1000.0);
	}

	public TransitionRealismAssessor getModelTRA() {
		return modelTRA;
	}

	public VariableTransitionApproximator getModelVTA() {
		return modelVTA;
	}
}
