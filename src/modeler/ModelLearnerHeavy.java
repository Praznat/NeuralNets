package modeler;

import java.util.*;

import reasoner.Foresight;

import utils.RandomUtils;
import deepnets.*;

/**
 * why model learning as opposed to policy learning?
 * with policy learning, if the reward function changes from training to test data youre screwed
 * with model learning, you can use the model to predict how to succeed in novel reward functions
 * @author alexanderbraylan
 *
 */
public class ModelLearnerHeavy extends ModelLearner {
	// TODO Weight pruning!!! Discretization & symbolization! (superstition reduction!)

//	private final TransitionFamiliarityAssessor modelTFA;
	private final VariableTransitionApproximator modelVTA;
	private final JointDistributionModeler modelJDM;
	
	public ModelLearnerHeavy(int errorHalfLife, int[] numHiddenVTA, int[] numHiddenTRA, int[] numHiddenJDM,
			ActivationFunction actFn, int maxReplaySize) {
		super(maxReplaySize);
		modelVTA = new VariableTransitionApproximator(actFn, numHiddenVTA, errorHalfLife);
//		modelTFA = new TransitionFamiliarityAssessor(actFn, numHiddenTRA, errorHalfLife);
		modelJDM = new JointDistributionModeler(actFn, numHiddenJDM, errorHalfLife);
	}
	
	@Override
	public void learnFromMemory(double lRate, double mRate, double sRate,
			boolean resample, int iterations, long displayProgressMs, double stopAtErrThreshold) {
		Collection<TransitionMemory> memories = experienceReplay.getBatch(resample);
		boolean debug = true;
		if (debug) System.out.println("Initiated Learning");
		long ms = System.currentTimeMillis();
		long lastMs = ms;
		for (int i = 0; i < iterations; i++) {
			for (TransitionMemory m : memories) getModelVTA().analyzeTransition(m, lRate, mRate, sRate);
			if (stopAtErrThreshold > 0 && getModelVTA().getConfidenceEstimate() < stopAtErrThreshold) break;
			if (displayProgressMs > 0 && System.currentTimeMillis() - lastMs >= displayProgressMs) {
				System.out.println(Utils.round(((double)i)*100 / iterations, 2) + "%");
				lastMs = System.currentTimeMillis();
			}
		}
		if (debug) ms = debugTime("Learning VTA took ", ms);
		lastMs = ms;
		for (int i = 0; i < iterations; i++) {
			for (TransitionMemory m : memories) getModelJDM().analyzeTransition(m, lRate, mRate, sRate);
			if (stopAtErrThreshold > 0 && getModelJDM().getConfidenceEstimate() < stopAtErrThreshold) break;
			if (displayProgressMs > 0 && System.currentTimeMillis() - lastMs >= displayProgressMs) {
				System.out.println(Utils.round(((double)i)*100 / iterations, 2) + "%");
				lastMs = System.currentTimeMillis();
			}
		}
		if (debug) ms = debugTime("Learning JDM took ", ms);
//		for (int i = 0; i < iterations; i++) {
//			for (TransitionMemory m : memories) getModelTFA().analyzeTransition(m, lRate, mRate, sRate);
//			if (stopAtErrThreshold > 0 && getModelTFA().getConfidenceEstimate() < stopAtErrThreshold) break;
//		}
//		if (debug) ms = debugTime("Learning TFA took ", ms);
		// TODO imagine one future using VTA, TFA learns familiarity based on how
		// close the imagined future is to the memory's postState
	}
	
	@Override
	public void feedForward() {
		TransitionMemory tm = createMemory();
		FFNeuralNetwork.feedForward(getModelVTA().getNeuralNetwork().getInputNodes(), tm.getPreStateAndAction());
//		if (workingPostState != null) {
//			FFNeuralNetwork.feedForward(getModelTFA().getNeuralNetwork().getInputNodes(), tm.getAllVars());
//			FFNeuralNetwork.feedForward(getModelJDM().getNeuralNetwork().getInputNodes(), tm.getAllVars());
//		}
	}

	@Override
	public double[] upJointOutput(double[] vars, int postStateIndex, int rounds) {
		vars = Foresight.probabilisticRoundingUnnormalized(vars); // adds noise
		Collection<? extends Node> inputNodes = modelJDM.getNeuralNetwork().getInputNodes();
		Collection<? extends Node> outputNodes = modelJDM.getNeuralNetwork().getOutputNodes();
		for (int i = 0; i < rounds; i++) {
			FFNeuralNetwork.feedForward(inputNodes, vars);
			int j = postStateIndex;
			for (Node n : outputNodes) vars[j++] = n.getActivation();
		}
		double[] result = new double[vars.length - postStateIndex];
		System.arraycopy(vars, postStateIndex, result, 0, result.length);
		return result;
	}
	public static double[] concatVars(double[] preState, double[] actions, double[] postState) {
		final int aLen = (actions != null ? actions.length : 0);
		final double[] vars = new double[preState.length + aLen + postState.length];
		System.arraycopy(preState, 0, vars, 0, preState.length);
		if (aLen > 0) System.arraycopy(actions, 0, vars, preState.length, aLen);
		System.arraycopy(postState, 0, vars, preState.length + aLen, postState.length);
		return vars;
	}
//	public double getFamiliarity(double[] allVars) {
//		final FFNeuralNetwork ann = getModelTFA().getNeuralNetwork();
//		FFNeuralNetwork.feedForward(ann.getInputNodes(), allVars);
//		return ann.getOutputNodes().get(0).getActivation();
//	}
//	@Override
//	public double[] upFamiliarity(TransitionMemory tm, int jointRounds, int epochs, double lRate, double mRate, double sRate, double maxShift) {
//		return upFamiliarity(tm.getAllVars(), tm.getPostState().length, jointRounds, epochs, lRate, mRate, sRate, maxShift);
//	}
//	@Override
//	public double[] upFamiliarity(double[] vars, int postStateLen, int jointRounds, int epochs, double lRate, double mRate, double sRate, double maxShift) {
//		FFNeuralNetwork ann = getModelTFA().getNeuralNetwork();
//		Collection<Connection> conns = getAllConnections(ann);
//		for (Connection conn : conns) conn.getWeight().frieze();
//
//		ArrayList<? extends Node> inputs = ann.getInputNodes();
//		ArrayList<Node> postStateInputs = new ArrayList<Node>();
//		int numInputs = inputs.size();
//		int postVarIndx = numInputs - postStateLen;
//		for (int i = postVarIndx; i < numInputs; i++) postStateInputs.add(inputs.get(i));
//		
//		vars = upJointOutput(vars, postVarIndx, jointRounds);
//		for (int epoch = 0; epoch < epochs; epoch++) {
//			moveInputsTowardOutputBP(ann, vars, postStateInputs, postVarIndx, lRate, mRate, sRate);
////			moveInputsTowardOutputST(ann, vars, postStateInputs, postVarIndx, maxShift);
//		}
//		
//		for (Connection conn : conns) conn.getWeight().unFrieze();
//		double[] result = new double[postStateLen];
//		System.arraycopy(vars, postVarIndx, result, 0, postStateLen);
//		return result;
//	}
	
	private static void moveInputsTowardOutputBP(FFNeuralNetwork ann, double[] vars, Collection<Node> postStateInputs,
			int postVarIndx, double lRate, double mRate, double sRate) {
		FFNeuralNetwork.feedForward(ann.getInputNodes(), vars);
		FFNeuralNetwork.backPropagate(ann.getOutputNodes(), lRate, mRate, sRate, 1);
		int i = postVarIndx;
		for (Node n : postStateInputs) {
			double nodeBlame = 0;
			for (Connection conn : n.getOutputConnections()) {
				final double blame = conn.getWeight().getBlameFromOutput();
				nodeBlame += blame;
			}
			final double a = vars[i];
			vars[i++] = Math.min(Math.max(a + nodeBlame, 0), 1); // not sure if should be += or -=
		}
	}
	private static void moveInputsTowardOutputST(FFNeuralNetwork ann, double[] vars, Collection<Node> postStateInputs,
			int postVarIndx, double maxShift) {
		int num = postStateInputs.size();
		for (int t = 0; t < num; t++) {
			final double shift = Math.random() * maxShift;
			final int i = postVarIndx + (int) (Math.random() * num);
			final double a = vars[i];
			final double up = Math.min(a + shift, 1);
			vars[i] = up;
			FFNeuralNetwork.feedForward(ann.getInputNodes(), vars);
			final double upOut = ann.getOutputNodes().get(0).getActivation();
			final double down = Math.max(a - shift, 0);
			vars[i] = down;
			FFNeuralNetwork.feedForward(ann.getInputNodes(), vars);
			final double downOut = ann.getOutputNodes().get(0).getActivation();
			if (upOut > downOut) vars[i] = up;
			// else vars[i] already = down
		}
		
		int i = postVarIndx;
		for (Node n : postStateInputs) {
			double nodeBlame = 0;
			for (Connection conn : n.getOutputConnections()) {
				final double blame = conn.getWeight().getBlameFromOutput();
				nodeBlame += blame;
			}
			vars[i++] += nodeBlame; // not sure if should be += or -=
		}
	}
	
	public static Collection<Connection> getAllConnections(FFNeuralNetwork ann) {
		Collection<Connection> result = new HashSet<Connection>();
		for (Layer<? extends Node> layer : ann.getLayers()) {
			for (Node n : layer.getNodes()) result.addAll(n.getOutputConnections());
		}
		return result;
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

//	public TransitionFamiliarityAssessor getModelTFA() {
//		return modelTFA;
//	}
	
	public JointDistributionModeler getModelJDM() {
		return modelJDM;
	}

	public VariableTransitionApproximator getModelVTA() {
		return modelVTA;
	}
	
	@Override
	public ModelerModule getTransitionsModule() {
		return getModelVTA();
	}

}
