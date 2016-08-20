package ann.indirectencodings;

import ann.ActivationFunction;
import ann.FFNeuralNetwork;

public class CPPN {
	
	private FFNeuralNetwork ann;
	
	public CPPN(RelationManager relManager, int[] numHidden) {
		this.ann = new FFNeuralNetwork(ActivationFunction.SIGMOID0p5, relManager.getInputSize(), 1, numHidden);
	}
	
	// TODO everything

}
