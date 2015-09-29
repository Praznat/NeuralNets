package reasoner;

import java.util.*;

import ann.*;
import modeler.ModelLearner;

public class Abduction {

	public static double[] backPropAbduction(ModelLearner modeler, double[] finalState,
			double[] fullInputStateGuess, int[] flexibleInputs) {
	
		FFNeuralNetwork ann = modeler.getTransitionsModule().getNeuralNetwork();
		ArrayList<? extends Node> inputNodes = ann.getInputNodes();
		
		Collection<Connection> conns = Connection.getAllConnections(ann);
		for (Connection conn : conns) conn.getWeight().frieze();
		
		double[] result = new double[fullInputStateGuess.length];
		System.arraycopy(fullInputStateGuess, 0, result, 0, fullInputStateGuess.length);
		
		int maxIterations = 10000;
		double lastAbsBlame = 1000;
		for (int i = 0; i < maxIterations; i++) {
			FFNeuralNetwork.feedForward(inputNodes, result);
			FFNeuralNetwork.backPropagate(ann.getOutputNodes(), 1.5, .5, 0, finalState);
			double maxAbsNodeBlame = 0;
			for (int k : flexibleInputs) {
				Node node = inputNodes.get(k);
				double nodeBlame = 0;
				for (Connection conn : node.getOutputConnections()) {
					final double blame = conn.getWeight().getBlameFromOutput();
					nodeBlame += blame;
				}
				result[k] = Math.min(Math.max(result[k] + nodeBlame, 0), 1); // not sure if should be += or -=
				maxAbsNodeBlame = Math.max(Math.abs(nodeBlame), maxAbsNodeBlame);
			}
			if (lastAbsBlame - maxAbsNodeBlame < 0.00001) break; // convergence
			lastAbsBlame = maxAbsNodeBlame;
		}
		for (Connection conn : conns) conn.getWeight().unFrieze();
		return result;
	}
}
