package sparsenets;

import java.util.Collection;

import ann.*;

public class SparseTests {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		testXORectifier(.05,2000);
		testXORSigmoid(1,100);
	}

	private static void testXORSigmoid(double lrate, int epochs) {
		FFNeuralNetwork rffn = new FFNeuralNetwork(ActivationFunction.SIGMOID0p5, 2, 1, new int[] {5});
		testXOR(rffn, lrate, epochs);
	}
	private static void testXORectifier(double lrate, int epochs) {
		FFNeuralNetwork rffn = new FFNeuralNetwork(ActivationFunction.RECTIFIER, 2, 1, new int[] {10});
		testXOR(rffn, lrate, epochs);
	}
	private static void testXORTreeSparse() {
		SparseFFNetwork sffn = new SparseFFNetwork(ActivationFunction.SIGMOID0p5, 2, 1, new int[] {15}, 2);
		testXOR(sffn, 1, 1000);
	}
	
	// use large hidden layer, make sure only n connections used
	private static void testXOR(NeuralNetwork nn, double learnRate, int epochs) {
		Collection<? extends Node> inputNodes = nn.getInputNodes();
		Collection<? extends Node> outputNodes = nn.getOutputNodes();
		double[][] inputSamples = {{0,0},{0,1},{1,0},{1,1}};
//		System.out.println("Untrained");
//		for (double[] is : inputSamples) {
//			FFNeuralNetwork.feedForward(inputNodes, is);
//			String output = "";
//			for (Node n : outputNodes) output += n.getActivation() + " ";
//			System.out.println(output);
//		}
		
		double[][] outputSamples = {{0},{1},{1},{0}};
		
		Collection<DataPoint> data = DataPoint.createData(inputSamples, outputSamples);
//		System.out.println("E: " + FFNeuralNetwork.stdError(inputNodes, outputNodes, data));

		long ms = System.currentTimeMillis();
		ControlPanel.learnFromBackPropagation(inputNodes, outputNodes, data, epochs,
				learnRate, learnRate, 0.9, 0.9, 0.005, 0.005);

//		System.out.println("Trained");
		for (double[] is : inputSamples) {
			FFNeuralNetwork.feedForward(inputNodes, is);
//			String output = "";
//			for (Node n : outputNodes) output += n.getActivation() + " ";
//			System.out.println(output);
		}
		System.out.println("E: " + FFNeuralNetwork.stdError(inputNodes, outputNodes, data));
		boolean success = FFNeuralNetwork.stdError(inputNodes, outputNodes, data) < 0.1;
		System.out.println((success ? "Success" : "Failure") + " in " + (System.currentTimeMillis() - ms) + " ms");
	}
}
