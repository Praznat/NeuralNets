package modeler;

import java.util.*;

import utils.*;
import deepnets.*;

public class VariableTransitionApproximator {
	
	private final ActivationFunction actFn;
	private final FFNeuralNetwork transitionANN;
	private final ExperienceReplay experienceReplay;
	private int numStateVars;
	private double[] inputActivations = {};
	private double[] outputTargets = {};
	private double errorEMA = 0;
	private final Decayer decayer;
	private int emaStep = 100;
	
	public VariableTransitionApproximator(int errorHalfLife, int[] numHidden,
			ActivationFunction actFn, int expReplaySize) {
		decayer = new Decayer(errorHalfLife);
		transitionANN = new FFNeuralNetwork(actFn, 0, 0, numHidden);
		if (actFn != ActivationFunction.SIGMOID0p5) throw new IllegalStateException("must use sigmoidal if representing probabilities");
		this.actFn = actFn;
		this.experienceReplay = expReplaySize > 0 ? new ExperienceReplay(expReplaySize) : null;
	}
	
	public FFNeuralNetwork getNeuralNetwork() {
		return transitionANN;
	}
	
	private void analyzeTransition(DataPoint dp, double lRate, double mRate, double sRate) {
		FFNeuralNetwork.feedForward(transitionANN.getInputNodes(), dp.getInputs());
		if(outputTargets[0] < 0) {
			System.out.println(outputTargets[0]);
		}
		final double error = FFNeuralNetwork.getError(dp.getOutputs(), transitionANN.getOutputNodes());
		errorEMA = decayer.newEMA(error, errorEMA, ++emaStep);
		FFNeuralNetwork.backPropagate(transitionANN.getOutputNodes(), lRate, mRate, sRate, dp.getOutputs());
		
	}
	
	public void observePreState(double... values) {
		numStateVars = values.length;
		observe(values, 0, numStateVars, 0);
	}
	public void observeAction(double... values) {
		observe(values, numStateVars, transitionANN.getLayerSize(0), 0);
	}
	public void observePostState(double... values) {
		numStateVars = values.length;
		observe(values, 0, numStateVars, transitionANN.getNumLayers() - 1);
	}
	
	private void observe(double[] values, int start, int end, int layer) {
		int layerSize = transitionANN.getLayerSize(layer);
		int shortage = values.length + start - layerSize;
		boolean inNotOut = layer == 0;
		if (shortage > 0) {
			for (int i = 0; i < shortage; i++) transitionANN.addNode(layer, actFn,
					inNotOut ? null : null);//new AccruingWeight(1.0));
			if (inNotOut) inputActivations = biggerArray(inputActivations, values.length + start);
			else outputTargets = biggerArray(outputTargets, values.length + start);
		}
		if (inNotOut) for (int i = start; i < end; i++) inputActivations[i] = values[i - start];
		else for (int i = start; i < end; i++) outputTargets[i] = values[i - start];
	}
	
	private double[] biggerArray(double[] array, int length) {
		double[] result = new double[length];
		System.arraycopy(array, 0, result, 0, array.length);
		return result;
	}

	public void saveMemory() {
		if (experienceReplay == null) throw new IllegalStateException("must define experience replay size");
		final DataPoint dp = DataPoint.create(inputActivations, outputTargets);
		experienceReplay.addMemory(dp);
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
			Collection<DataPoint> memories = batchSize > 0 ? experienceReplay.getBatch(batchSize, resample)
					: experienceReplay.getBatch(resample);
			for (DataPoint m : memories) {
				analyzeTransition(m, lRate, mRate, sRate);
				// print out I/O for debugging
//				String s = "";
//				for (double d : m.getInputs()) s += d+"	";
//				s += "	:	";
//				for (Node n : transitionANN.getOutputNodes()) s += n.getActivation()+"	";
//				System.out.println(s);
			}
//			System.out.println(this.getConfidenceEstimate());
			if (stopAtErrThreshold > 0 && this.getConfidenceEstimate() < stopAtErrThreshold) return;
		}
	}
	public void learn(double lRate, double mRate, double sRate) {
		final DataPoint dp = new DataPoint(inputActivations, outputTargets);
		analyzeTransition(dp, lRate, mRate, sRate);
	}
	public void feedForward() {
		FFNeuralNetwork.feedForward(transitionANN.getInputNodes(), inputActivations);
	}
	
	public double getConfidenceEstimate() {
		return errorEMA;
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
				FFNeuralNetwork.feedForward(transitionANN.getInputNodes(), inputActivations);
				int i = 0;
				for (Node n : transitionANN.getOutputNodes()) outputActivation[i++] = n.getActivation();
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
	
	public double[] getInputActivations() {
		return inputActivations;
	}
	
	private static String r(double activation) {
		return String.valueOf(((int)Math.round(activation*1000))/1000.0);
	}
}
