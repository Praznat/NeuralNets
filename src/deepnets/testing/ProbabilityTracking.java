package deepnets.testing;

import java.util.*;

import modeler.*;
import reasoner.DiscreteState;
import deepnets.*;

public class ProbabilityTracking {
	

	public static void main(String[] args) {
		
		int vectSize = 8;
		int mapSize = 5;
		int sampleSize = 100;
		int learnIterations = 1000;
		
		Map<DiscreteState,Double> map = createMap(vectSize, mapSize);
		Collection<DataPoint> data = createData(map, sampleSize);
		
		ModelLearner modeler = new ModelLearnerHeavy(500, new int[] {vectSize*3},
				null, new int[] {vectSize * 2}, ActivationFunction.SIGMOID0p5, data.size());
		
		for (DataPoint datum : data) {
			modeler.observePreState(datum.getInput());
			modeler.observePostState(datum.getOutput());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(0.9, 0.8, 0, false, learnIterations, 10000);
		for (Map.Entry<DiscreteState,Double> entry : map.entrySet()) {
			modeler.observePreState(entry.getKey().getRawState());
			modeler.feedForward();
			double pred = modeler.getTransitionsModule().getOutputActivations()[0];
			System.out.println(entry.getKey() + "	" + entry.getValue() + "	" + pred);
		}
	}
	

	private static Collection<DataPoint> createData(Map<DiscreteState,Double> map, int sampleSize) {
		Collection<DataPoint> result = new ArrayList<DataPoint>();
		for (Map.Entry<DiscreteState,Double> entry : map.entrySet()) {
			for (Integer i : createSamples(entry.getValue(), sampleSize)) {
				result.add(new DataPoint(entry.getKey().getRawState(), new double[] {i}));
			}
		}
		return result;
	}
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
	
	private static ArrayList<Integer> createSamples(double p, int n) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		int cutoff = (int) Math.round(n * p);
		for (int i = 0; i < cutoff; i++) result.add(1);
		for (int i = cutoff; i < n; i++) result.add(0);
		return result;
	}
}
