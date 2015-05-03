package modeler;

import java.util.Collection;

import deepnets.*;

/**
 * why model learning as opposed to policy learning?
 * with policy learning, if the reward function changes from training to test data youre screwed
 * with model learning, you can use the model to predict how to succeed in novel reward functions
 * @author alexanderbraylan
 *
 */
public class ModelLearnerLite extends ModelLearner {
	// TODO Weight pruning!!! Discretization & symbolization! (superstition reduction!)

	private final JointAndConditionalModeler modelJCM;
	
	public ModelLearnerLite(int errorHalfLife, int[] numHiddenVTA, int[] numHiddenTRA, int[] numHiddenJDM,
			ActivationFunction actFn, int expReplaySize) {
		super(expReplaySize);
		modelJCM = new JointAndConditionalModeler(actFn, numHiddenJDM, errorHalfLife);
	}
	
	@Override
	public void learnOnline(double lRate, double mRate, double sRate) {
		TransitionMemory m = saveMemory();
		getModelJCM().analyzeTransition(m, lRate, mRate, sRate);
	}
	
	@Override
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, long displayProgressMs, double stopAtErrThreshold) {
		Collection<TransitionMemory> memories = experienceReplay.getBatch();
		boolean debug = true;
		if (debug) System.out.println("Initiated Learning");
		long ms = System.currentTimeMillis();
		for (int i = 0; i < iterations; i++) {
			for (TransitionMemory m : memories) getModelJCM().analyzeTransition(m, lRate, mRate, sRate);
			if (stopAtErrThreshold > 0 && getModelJCM().getConfidenceEstimate() < stopAtErrThreshold) break;
		}
		if (debug) ms = debugTime("Learning JCM took ", ms);
	}
	
	@Override
	public void feedForward() {
		TransitionMemory tm = createMemory();
		FFNeuralNetwork.feedForward(getModelJCM().getNeuralNetwork().getInputNodes(), tm.getPreStateAndAction());
		if (workingPostState != null) FFNeuralNetwork.feedForward(getModelJCM().getNeuralNetwork().getInputNodes(), tm.getAllVars());
	}

	@Override
	public double[] upJointOutput(double[] vars, int postStateIndex, int rounds) {
		return vars; // TODO THIS NEEDS TO BE JUST POST VARS!
	}
//	@Override
//	public double[] upFamiliarity(TransitionMemory tm, int jointRounds, int epochs, double lRate, double mRate, double sRate, double maxShift) {
//		return upFamiliarity(tm.getAllVars(), tm.getPostState().length, jointRounds, epochs, lRate, mRate, sRate, maxShift);
//	}
//	@Override
//	public double[] upFamiliarity(double[] vars, int postStateLen, int jointRounds, int epochs, double lRate, double mRate, double sRate, double maxShift) {
//		return upJointOutput(vars, vars.length - postStateLen, jointRounds);
//	}
//	@Override
//	public double getFamiliarity(double[] allVars) {
//		return 1;
//	}
	
	protected static String r(double activation) {
		return String.valueOf(((int)Math.round(activation*1000))/1000.0);
	}

	public JointAndConditionalModeler getModelJCM() {
		return modelJCM;
	}
	
	@Override
	public ModelerModule getFamiliarityModule() {
		return getModelJCM();
	}
	@Override
	public ModelerModule getTransitionsModule() {
		return getModelJCM();
	}

}
