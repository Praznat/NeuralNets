package deepnets.testing;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import modeler.*;
import reasoner.*;
import deepnets.*;

public class ProbabilityTracking {
	
	/**
	 * ordering: p'(Y|X) can be said to have correct ordering if:
	 * for all vectors i, j, and random variables X, Y,
	 * p(Y=i|X=i) > p(Y=j|X=j) -> p'(Y=i|X=i) > p'(Y=j|X=j)
	 * where p(Y|X) is TRUE conditional probability
	 * and p'(Y|X) is ESTIMATED conditional probability
	 */

	public static void main(String[] args) {
//		test1dOutput();
		testMultimensionalOutput();
	}
	
	private static ModelLearner createModelLearner(int vectSize, Collection<DataPoint> data) {
		ModelLearner modeler = new ModelLearnerHeavy(500, new int[] {vectSize + 1},
				null, new int[] {vectSize * vectSize + 1}, ActivationFunction.SIGMOID0p5, data.size());
		
		for (DataPoint datum : data) {
			modeler.observePreState(datum.getInput());
			modeler.observePostState(datum.getOutput());
			modeler.saveMemory();
		}
		int learnIterations = 250;
		modeler.learnFromMemory(0.9, 0.6, 0, false, learnIterations, 10000);
		System.out.println();
		return modeler;
	}
	
	private static double[] getModelerPrediction(ModelLearner modeler, double[] input) {
		modeler.observePreState(input);
		modeler.feedForward();
//		double pred = modeler.getTransitionsModule().getOutputActivations()[0];
		double[] gpred = Foresight.montecarlo(modeler, input, null, 1, 50, 8);
		return gpred;
	}
	
	public static void test1dOutput() {
		
		int vectSize = 3;
		int mapSize = 5;
		int sampleSize = 100;
		
		Map<DiscreteState,Double> map = createMap(vectSize, mapSize);
		Collection<DataPoint> data = createData(map, sampleSize);
		
		ModelLearner modeler = createModelLearner(vectSize, data);
		for (Map.Entry<DiscreteState,Double> entry : map.entrySet()) {
			double[] pred = getModelerPrediction(modeler, entry.getKey().getRawState());
			System.out.println(entry.getKey() + "	" + entry.getValue() + "	" + pred);
		}
	}
	
	/** creates random data */
	private static Collection<DataPoint> createData(Map<DiscreteState,Double> map, int sampleSize) {
		Collection<DataPoint> result = new ArrayList<DataPoint>();
		for (Map.Entry<DiscreteState,Double> entry : map.entrySet()) {
			for (double[] o : createSamples(entry.getValue(), sampleSize)) {
				result.add(new DataPoint(entry.getKey().getRawState(), o));
			}
		}
		return result;
	}
	
	/** creates map of vectors and corresponding 1-d probabilities */
	private static Map<DiscreteState,Double> createMap(int vectSize, int mapSize) {
		Map<DiscreteState,Double> map = new HashMap<DiscreteState,Double>();
		int n = (int) Math.min(mapSize, Math.pow(2, vectSize));
		while (map.size() < n) {
			double[] addition = new double[vectSize];
			for (int i = 0; i < vectSize; i++) addition[i] = Math.round(Math.random());
			map.put(new DiscreteState(addition), Math.random());
		}
		return map;
	}
	
	/** makes list of n size-1 outputs given probability p */
	private static ArrayList<double[]> createSamples(double p, int n) {
		ArrayList<double[]> result = new ArrayList<double[]>();
		int cutoff = (int) Math.round(n * p);
		for (int i = 0; i < cutoff; i++) result.add(new double[] {1});
		for (int i = cutoff; i < n; i++) result.add(new double[] {0});
		return result;
	}
	
	// TODO enforce correlations
	// TODO test decision process like this on BasicTests toy problem
	public static void testMultimensionalOutput() {
		int vectSize = 8;
		int outputsPerInput = 10;
		double repeatProb = 0.8;
		double inpRemovalPct = 0.8;
//		Collection<DataPoint> data = createMultidimensionalCorrSamples(vectSize, outputsPerInput, inpRemovalPct);
		Collection<DataPoint> data = createMultidimensionalSamples(vectSize, outputsPerInput, repeatProb);

		ModelLearner modeler = createModelLearner(vectSize, data);
		int numRuns = 100;
		int jointAdjustments = 18;
		double skewFactor = 0;
		double cutoffProb = 0.1;
		DecisionProcess decisioner = new DecisionProcess(modeler, null, 1, numRuns,
				0, skewFactor, 0, cutoffProb);
		DecisionProcess decisionerJ = new DecisionProcess(modeler, null, 1, numRuns,
				jointAdjustments, skewFactor, 0, cutoffProb);
		
		Set<DiscreteState> inputs = getInputSetFromSamples(data);
		ArrayList<Double> realV = new ArrayList<Double>();
		ArrayList<Double> predV = new ArrayList<Double>();
		ArrayList<Double> predJV = new ArrayList<Double>();
		for (DiscreteState input : inputs) {
			System.out.println("S" + input);
			Map<DiscreteState, Double> outProbs = getRealOutputProbsForInput(input, data);
			Map<DiscreteState,Double> preds = countToFreqMap(decisioner
					.getImmediateStateGraphForActionGibbs(input.getRawState(), new double[] {}));
			Map<DiscreteState,Double> predsJ = countToFreqMap(decisionerJ
					.getImmediateStateGraphForActionGibbs(input.getRawState(), new double[] {}));
			Set<DiscreteState> outputs = new HashSet<DiscreteState>();
			outputs.addAll(outProbs.keySet());
			outputs.addAll(preds.keySet());
			outputs.addAll(predsJ.keySet());
			for (DiscreteState output : outputs) {
				Double realD = outProbs.get(output);
				Double predD = preds.get(output);
				Double predJD = predsJ.get(output);
				double real = (realD != null ? realD : 0);
				double pred = (predD != null ? predD : 0);
				double predJ = (predJD != null ? predJD : 0);
				realV.add(real);
				predV.add(pred);
				predJV.add(predJ);
				System.out.println("	S'" + output + "	" + real + "	" + pred + "	" + predJ);
			}
		}
		System.out.println("CORR:	" + Utils.correlation(realV, predV)
				+ "	" + Utils.correlation(realV, predJV));
	}
	
	private static Collection<DataPoint> createMultidimensionalSamples(int vectSize,
			int outputsPerInput, double repeatProb) {
		Collection<DataPoint> result = new ArrayList<DataPoint>();
		int numSamples = vectSize * vectSize;
		for (int s = 0; s < numSamples; s++) {
			double[] input = new double[vectSize];
			for (int i = 0; i < vectSize; i++) input[i] = Math.round(Math.random());
			DataPoint dp = null;
			for (int j = 0; j < outputsPerInput; j++) {
				double[] output = new double[vectSize];
				for (int i = 0; i < vectSize; i++) output[i] = Math.round(Math.random());
				if (Math.random() > repeatProb || j == 0) dp = new DataPoint(input, output);
				result.add(dp);
			}
		}
		return result;
	}
	
	private static Collection<DataPoint> createMultidimensionalCorrSamples(int vectSize,
			int numSamplesPerInput, double inputRemovalPct) {
		Collection<DataPoint> result = new ArrayList<DataPoint>();
		List<double[]> inputs = possibleInputs(vectSize);
		Collections.shuffle(inputs);
		int remo = (int) (inputRemovalPct * inputs.size());
		for (int i = 0; i < remo; i++) inputs.remove(0);
		for (double[] input : inputs) {
			double[][] m = randomSqrMatrix(vectSize);
			Collection<double[]> outs = createCorrelatedOutputs(vectSize, numSamplesPerInput, m);
			for (double[] output : outs) result.add(new DataPoint(input, output));
		}
		return result;
	}
	
	private static List<double[]> possibleInputs(int vectSize) {
		List<double[]> result = new ArrayList<double[]>();
		for (int i = 0; i < vectSize * vectSize; i ++) {
			String s = String.format("%" + vectSize + "s", Integer.toBinaryString(i)).replace(' ', '0');
			double[] r = new double[vectSize];
			for (int j = 0; j < vectSize; j++) r[j] = Integer.parseInt(s.substring(j, j+1));
			result.add(r);
		}
		return result;
	}
	
	private static Collection<double[]> createCorrelatedOutputs(int vectSize, int num, double[][] m) {
		Collection<double[]> result = new ArrayList<double[]>();
		List<Integer> ord = new ArrayList<Integer>();
		for (int i = 0; i < vectSize; i++) ord.add(i);
		for (int s = 0; s < num; s++) {
			Collections.shuffle(ord);
			double[] out = new double[vectSize];
			int lastK = ord.get(0);
			out[lastK] = Math.round(Math.random());
			for (int i = 1; i < ord.size(); i++) {
				int k = ord.get(i);
				double corr = getFromCorrM(m, k, lastK);
				if (Math.random() < Math.abs(corr)) out[k] = corr > 0 ? out[lastK] : (1-out[lastK]);
				else out[k] = Math.round(Math.random());
				result.add(out);
				lastK = k;
			}
		}
		return result;
	}
	
	private static double[][] randomSqrMatrix(int size) {
		double[][] m = new double[size][size];
		for (int i = 0; i < size; i++) for (int j = 0; j < size; j++) m[i][j] = Math.round(Math.random() * 2 - 1);
		return m;
	}
	
	private static double getFromCorrM(double[][] m, int i, int j) {
		if (i < j) return m[i][j];
		else return m[j][i];
	}


	private static Set<DiscreteState> getInputSetFromSamples(Collection<DataPoint> data) {
		Set<DiscreteState> result = new HashSet<DiscreteState>();
		for (DataPoint dp : data) result.add(new DiscreteState(dp.getInput()));
		return result;
	}
	
	private static Map<DiscreteState, Double> getRealOutputProbsForInput(DiscreteState input,
			Collection<DataPoint> data) {
		Map<DiscreteState, AtomicInteger> counts = new HashMap<DiscreteState, AtomicInteger>();
		for (DataPoint dp : data) {
			if ((new DiscreteState(dp.getInput())).equals(input)) {
				DiscreteState output = new DiscreteState(dp.getOutput());
				AtomicInteger count = counts.get(output);
				if (count == null) counts.put(output, count = new AtomicInteger());
				count.incrementAndGet();
			}
		}
		return countToFreqMap(counts);
	}
	
	private static Map<DiscreteState, Double> countToFreqMap(Map<DiscreteState, AtomicInteger> counts) {
		int sum = 0;
		for (AtomicInteger ai : counts.values()) sum += ai.get();
		Map<DiscreteState, Double> result = new HashMap<DiscreteState, Double>();
		for (Map.Entry<DiscreteState, AtomicInteger> entry : counts.entrySet()) {
			result.put(entry.getKey(), entry.getValue().doubleValue() / sum);
		}
		return result;
	}
}
